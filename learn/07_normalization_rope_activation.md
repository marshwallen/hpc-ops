# 07 - Normalization、RoPE 与 Activation 融合

> Transformer 里的"小算子":RMSNorm、RoPE 旋转位置编码、SiLU 激活、量化。每个单看都很简单(几行 PyTorch),但它们**位于 hot path**,组合起来就成为延迟瓶颈。HPC-Ops 把每组相邻的小算子做成单个 CUDA kernel,中间结果不出 register。

## 1. 算子家族概览

```
Normalization:
  fused_rmsnorm_with_scale     ← RMSNorm + 量化到 FP8

RoPE:
  rope_norm_store_kv           ← BF16: split QKV + RoPE + (可选)QK-Norm + 存 paged KV
  rope_norm_store_kv_fp8       ← 同上 FP8 版

Activation:
  act_mul_and_quant            ← SwiGLU + FP8 量化 (per-tensor scale)
  masked_act_mul_and_quant     ← 同上,带 expert mask
  masked_act_mul_and_blockwise_quant  ← 同上,blockwise scale
  scaled_fp8_quant             ← 独立量化算子(per-tensor scale)
```

每个的 Python 接口在 `hpc/normalization.py` / `hpc/rope.py` / `hpc/act.py`,CUDA 实现在 `src/normalization/` / `src/rope/` / `src/activation/`。

## 2. RMSNorm + FP8 量化融合

### 2.1 标准 RMSNorm

```python
def rmsnorm(x, weight, eps):
    # x: [..., H]
    rms = torch.sqrt((x.float() ** 2).mean(-1, keepdim=True) + eps)
    return (x / rms) * weight
```

`x` 沿最后一维归一化,再乘以可学习权重 `weight: [H]`。

### 2.2 RMSNorm + FP8 量化

下游 GEMM 走 FP8,所以 RMSNorm 之后通常紧接一个量化:

```python
y_bf16 = rmsnorm(x, weight, eps)
y_fp8  = (y_bf16 / scale).to(torch.float8_e4m3fn)
```

朴素实现:
- `rmsnorm` kernel:H 维度 reduce + elementwise,写出 bf16 中间张量
- `quant` kernel:读 bf16,写 fp8

**融合后**:reduce 完后 inv_rms 留在 register,elementwise 一次性完成 normalize + scale + cast to fp8。中间无 bf16 张量。

### 2.3 MoE 场景:同时输出多种格式

```python
hpc.fused_rmsnorm_with_scale(
    a,           # [B, H] bf16
    weight,      # [H] bf16
    eps,
    scale,       # [1] 或 [2] fp32
    is_moe=False,
)
```

- `is_moe=False`:返回 `fp8_output = rmsnorm(a) / scale[0]` 一个张量
- `is_moe=True`:返回三个张量
  - `fp32_output = rmsnorm(a)`(MoE 的共享路径或精度敏感分支)
  - `fp8_output_1 = rmsnorm(a) / scale[0]`(给 router GEMM)
  - `fp8_output_2 = rmsnorm(a) / scale[1]`(给 down GEMM,通常和 scale[0] 不同)

一次 RMSNorm 输出三种用途,**reduce 只做一次**。

### 2.4 限制

源码 entry 里约束:
- `input` 和 `weight` 必须是 `bfloat16`
- hidden_size 必须是 `5120 / 4096 / 320` 之一(其他需要新增 dispatch case)

## 3. RoPE + Norm + KV Store 融合

### 3.1 RoPE 是什么

Rotary Positional Embedding,把位置编码"旋转"进 Q 和 K 的偶/奇维度:

```python
def rope(x, cos, sin):
    # x: [..., D]
    # cos, sin: [seq_len, D]
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos
    return interleave(out_even, out_odd)
```

实际实现里 cos/sin 通常预算好放在 `cos_sin: [max_seq, head_dim]`。

### 3.2 Attention 前的常见处理流

```
qkv = MatMul(x, W_qkv)              # [N, total_dim]
                                    # total_dim = num_q*qk_dim + num_kv*qk_dim + num_kv*v_dim
分解 -> q[N, H_q, D_qk], k[N, H_kv, D_qk], v[N, H_kv, D_v]
↓
(可选)QK RMSNorm
↓
RoPE 旋转(只对 q, k,不对 v)
↓
把新的 k, v 写入 paged KV cache 的对应槽位
↓
返回 q(供后续 attention 用)
```

**5 步,5 个独立 kernel** 是朴素做法。

### 3.3 HPC-Ops 的融合算子

```python
out_q = hpc.rope_norm_store_kv(
    key_cache,             # [num_blocks, block_size, num_kv_heads, qk_head_dim]
    value_cache,           # [num_blocks, block_size, num_kv_heads, v_head_dim]
    qkv,                   # [num_rows, total_dim] packed
    cos_sin,               # [max_seq, qk_head_dim] fp32 预算的 cos/sin 表
    num_seqlen_per_req,    # [num_req] int32  每条请求当前总长度
    q_index,               # [num_req + 1] int32  Q 的前缀和索引
    kvcache_indices,       # [num_req, max_blocks] int32  paged block 表
    is_prefill,            # bool
    q_norm_weight=None,    # 可选 RMSNorm 权重 for Q
    k_norm_weight=None,
    qk_norm_policy=0,      # 0=不做 norm, 1=先 RoPE 后 norm, 2=先 norm 后 RoPE
)
```

返回 `out_q: [num_rows, num_q_heads, qk_head_dim]`。**K 和 V 直接 in-place 写入 cache**。

### 3.4 FP8 版本

```python
out_q_fp8, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
    ...
    k_scale,           # [1] static k scale
    v_scale,           # [1] static v scale
    quant_policy,      # 1=dqskv 动态 per-token per-head Q scale; 2=sqskv 静态
    max_seqlens,       # int (prefill 时用于 q_scale 分配大小)
    upper_max=None,    # FP8 饱和上界(默认 ~448)
    q_scale_inv=None,  # quant_policy=2 时必填
    ...
)
```

返回:
- `out_q_fp8`:量化后的 Q
- `q_scale`:动态量化时 kernel 计算出的 scale
- `split_k_flag`:被 zeroed 出来供下游 attention 用

### 3.5 为什么把这么多事塞一个 kernel

每个 qkv 行只**读一次 HBM**,然后:
- 分解
- (可选)norm
- rope
- 量化(fp8 路径)
- 输出 q
- 输出 k 到 cache 槽
- 输出 v 到 cache 槽

寄存器内完成,**1 次 HBM 读 + 3 次 HBM 写**(最优)。朴素实现可能是 5 次读 5 次写。

### 3.6 `qk_norm_policy` 三种语义

| policy | 顺序 | 用途 |
|---|---|---|
| 0 | 不做 norm | 标准模型 |
| 1 | RoPE → RMSNorm | DeepSeek-V3 等 |
| 2 | RMSNorm → RoPE | 其他变体 |

通过这个开关一个 kernel 覆盖多种模型架构。

## 4. SwiGLU + FP8 量化融合

### 4.1 SwiGLU

```
gate_up: [N, 2C]   (concat 的 gate 和 up 投影)
gate = gate_up[:, :C]
up   = gate_up[:, C:]
silu_gate = gate * sigmoid(gate)      # SiLU
gated     = silu_gate * up
```

### 4.2 融合到 FP8

```python
y_fp8 = hpc.act_mul_and_quant(
    gate_up,         # [N, 2C] bf16
    scale,           # [1] fp32 量化 scale
    use_bf16_mul=True,
    output=None,
)
# y_fp8: [N, C] fp8_e4m3fn
```

内部:

```cuda
__global__ void act_mul_and_quant_kernel(...) {
    int row = blockIdx.x;
    int col_base = threadIdx.x * vec_size;
    
    // 向量 load 2C 个 bf16
    bfloat16 gate_vec[V] = load(gate_up + row * 2C + col_base);
    bfloat16 up_vec[V]   = load(gate_up + row * 2C + C + col_base);
    
    // 计算 silu(gate) * up
    for (int i = 0; i < V; ++i) {
        float g = float(gate_vec[i]);
        float u = float(up_vec[i]);
        float silu_g = g / (1.0f + expf(-g));
        float result = silu_g * u * scale_val;
        // 饱和到 fp8 范围
        result = clamp(result, -fp8_max, fp8_max);
        out_vec[i] = fp8_e4m3(result);
    }
    
    // 向量 store
    store(output + row * C + col_base, out_vec);
}
```

### 4.3 Masked 版本(MoE 用)

`masked_act_mul_and_quant` 多接一个 `num_per_expert: [num_expert] int32`。kernel 内部布局:

```
gate_up:    [num_expert × num_token_padded_per_expert, 2C]
            └─ padded:每个 expert 都按最大 token 数 padding
num_per_expert: 每个 expert 的真实 token 数

只对 [0..num_per_expert[e]) 范围内的 token 做计算,
其他位置写 0(或不写)
```

避免对 padded 区域做无用计算。

### 4.4 Blockwise quant 变体

`masked_act_mul_and_blockwise_quant` 用 blockwise scale(每 128 元素一个),输出多一个 `output_scale: [N, C/128] fp32`。算法:

```
对每 128 个元素一组:
  1. 算出 silu(gate)*up,留在 register
  2. 找这 128 个值的 max(abs(.))
  3. scale = max / fp8_max
  4. fp8_out = values / scale
  5. 写出 fp8_out (128 个 fp8) 和 scale (1 个 fp32)
```

每组只扫一次 register,精度比 per-tensor 高一档。

## 5. 独立的 `scaled_fp8_quant`

最朴素的算子:`y_fp8 = clamp(x / scale, -fp8_max, fp8_max).to(fp8_e4m3fn)`。

```python
output, scale = hpc.scaled_fp8_quant(input, scale, output=None)
```

支持 input dtype 为 `float32 / float16 / bfloat16`,scale 必须是 `[1] float32`。给上层做"准备 fp8 输入"用,无任何融合,仅是一个高吞吐的 fused vec load + cast + store。

## 6. 数据布局共性

所有这些算子的共同假设:
- 输入张量**最后一维 contiguous**(`stride(-1) == 1`)
- hidden_size / intermediate_size 是 8 或 128 的倍数(`float4` 向量化要求)
- per-tensor scale 必须是 `[1]` 形状的 fp32 张量(不接受 Python scalar,因为要 device-resident)

如果违反,kernel 会通过 `TORCH_CHECK` 报错并提示需要的形状。

## 7. 阅读源码路径

1. **`hpc/normalization.py`, `hpc/rope.py`, `hpc/act.py`**:Python 接口,docstring 描述各张量布局。
2. **`src/normalization/entry.cc`**:看 wrapper 验证逻辑。
3. **`src/normalization/fused_rmsnorm_with_scale.cu`**:RMSNorm + FP8 量化的 kernel(注意 hidden_size 编译期 dispatch)。
4. **`src/rope/rope.cu`**:RoPE + KV Store kernel,重点看 cos/sin 表查 + paged block 索引计算。
5. **`src/activation/activation.cu`**:SwiGLU + 量化 kernel,重点看 vector load + warp 内向量化处理。

## 8. 测试

每个算子都有独立 pytest:
- `tests/test_normalization.py`
- `tests/test_rope.py`
- `tests/test_act.py`
- `tests/test_attention_with_kvcache_*.py`(间接测 RoPE+KV Store)

测试结构通用:
1. 构造 PyTorch 参考实现(显式写 RMSNorm/SwiGLU)
2. 跑 HPC-Ops
3. `allclose(ref, my, atol=...)` 验证

---

下一篇:[08_stem_sparse_attention.md](./08_stem_sparse_attention.md) — Stem 块稀疏注意力的评分管线。
