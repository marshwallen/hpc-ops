# 04 - Fused MoE

> MoE(Mixture of Experts)是当下大模型的主流架构:DeepSeek-V3、Hunyuan-V3、Qwen3-235B 等都是 MoE。MoE 推理的难点不在单个 expert 计算,而在**路由**和**多 stage 串联**带来的 launch overhead 和 IO 浪费。HPC-Ops 的 Fused MoE 把整条 MoE 路径压成 1-3 个 kernel。

## 1. MoE 速览

每个 MoE 层的标准计算:

```
        x: [N, hidden]               ← 输入 N 个 token
        │
        ▼
   Router (linear + softmax)         ← 给每个 token 选 top-k 个 expert
        │
        ├─→ topk_ids[N, k]           ← 每个 token 选中的 expert id
        └─→ topk_scale[N, k]         ← 对应的权重(softmax 输出)
        │
        ▼ (分发 token 到对应 expert)
   For expert in 0..E-1:
     m_e = 收到的 token 数(变长!)
     x_e: [m_e, hidden]
     │
     ▼  Gate-Up GEMM (双层 FFN 的前半)
     gate_up_e = x_e @ W_gateup_e^T            # [m_e, 2 * intermediate]
     │
     ▼  Activation: SwiGLU = silu(gate) * up
     gated_e = silu(gate_up_e[:, :inter]) * gate_up_e[:, inter:]   # [m_e, inter]
     │
     ▼  Down GEMM (双层 FFN 的后半)
     out_e = gated_e @ W_down_e^T              # [m_e, hidden]
        │
        ▼ (按 topk_scale 加权回填到原 token 位置)
   y[i] = sum_k (out_{topk[i,k]} * topk_scale[i,k])
```

朴素实现需要的 kernel 数:

```
1. Router          (linear)
2. topk             (排序)
3. Gather          (按 expert 重排 token)
4. Gate-Up GEMM    (E 个或 1 个 grouped GEMM)
5. SiLU + mul + quant
6. Down GEMM       (E 个或 1 个 grouped GEMM)
7. Scatter + reduce
```

**7 个 kernel,7 次中间结果 HBM 往返**。

HPC-Ops 的目标:把这条路径**压成 2-3 个 kernel**,Gate-Up 和 Down 都用自研 grouped GEMM,前后接 cp.async pipeline,中间用 PDL 串起来。

## 2. Python API

`hpc/fuse_moe.py` 暴露三个主要算子:

### 2.1 per-tensor FP8 版本

```python
y = hpc.fuse_moe(
    x,                    # [N, hidden] fp8
    gate_up_weight,       # [E, 2*intermediate, hidden] fp8
    down_weight,          # [E, hidden, intermediate] fp8
    gate_up_scale,        # [E] fp32  per-expert scale
    down_scale,           # [E] fp32
    act_and_mul_scale,    # [1] fp32  激活后的 fp8 量化 scale
    topk_ids,             # [N, k] int32  ← 上游 router 给的
    topk_scale,           # [N, k] fp32   ← 同上
    rank_ep,              # 当前 EP rank
    num_expert_total,     # 全局 expert 数
    use_bf16_mul=True,
    shared_output=None,   # 可选:共享专家的输出(MoE+共享专家时用)
)
# y: [N, hidden] bf16
```

### 2.2 blockwise FP8 版本

```python
y = hpc.fuse_moe_blockwise(
    x,                          # [N, hidden] fp8
    x_scale,                    # blockwise fp32 scale
    gate_up_weight, gate_up_weight_scale,   # blockwise
    down_weight, down_weight_scale,         # blockwise
    topk_ids, topk_scale,
    rank_ep, num_expert_total,
    shared_output=None, output=None,
)
```

### 2.3 底层辅助算子

```python
# 1. Routing + 排序 + 分配(返回 8 个张量)
gate_up_input, gate_up_output, topk_pos, seqlens, cu_seqlens, \
tiles, cu_tiles, tmas = hpc.count_and_gather(
    x, topk_ids, num_expert, rank_ep, intermediate_size, num_seq_per_group_avg
)

# 2. 最后的加权 scatter-add
y = hpc.reduce(x, topk_pos, topk_scale, shared_output=None)
```

通常用户**只调高层的 `fuse_moe`**,但理解底层算子有助于知道 fuse_moe 内部在做什么。

## 3. count_and_gather:Routing 的高效实现

### 3.1 朴素 routing 的问题

最直接的实现:

```python
for i in range(N):
    for j in range(k):
        e = topk_ids[i, j]
        # 把 x[i] copy 到 expert e 的 token 列表里
        idx = atomicAdd(seqlens[e], 1)
        expert_buffer[e][idx] = x[i]
```

问题:**E 路 atomicAdd 抢同一个计数器**,在 H100 上 E=128 时,这一步可能比 GEMM 还慢。

### 3.2 HPC-Ops 的做法

源码 `src/fuse_moe/count_and_gather.cu`:

```
Phase 1: 用 shared memory 做线程块内 counting
    每个 CTA 处理 N/numCTA 个 token
    在 SMEM 里维护 [E] 个计数器(thread-private)
    对每个 token 的 k 个 expert id 做 SMEM atomicAdd
    最后 reduce 整 CTA 的计数器,写回 [numCTA, E] 的全局 partial counts

Phase 2: 跨 CTA prefix sum
    扫一遍全局 partial counts 得到每个 expert 的总 token 数 seqlens[e]
    + 累加得到 cu_seqlens

Phase 3: 第二次遍历,按 cu_seqlens 直接写入连续区段
    expert_offset = cu_seqlens[e] + counter_local[e]
    expert_buffer[expert_offset] = x[i]
    counter_local[e] += 1
```

每个 token 只有 SMEM atomic,全局 atomic 几乎没有,**吞吐线性 scale**。

### 3.3 输出张量速览

`count_and_gather` 返回 8 个张量:

| 张量 | 形状 | 含义 |
|---|---|---|
| `gate_up_input` | `[N*k, hidden]` fp8 | 按 expert 重排的 token,供 Gate-Up GEMM 读 |
| `gate_up_output` | `[N*k, intermediate]` bf16 | 预分配的 Gate-Up GEMM 输出缓冲 |
| `topk_pos` | `[N, k]` int32 | 每个 token 在排序后的位置(给 reduce 反向用)|
| `seqlens` | `[E]` int32 | 每个 expert 收到多少 token |
| `cu_seqlens` | `[E+1]` int32 | seqlens 的前缀和 |
| `tiles, cu_tiles` | `[E], [E+1]` | grouped GEMM 的 tile 划分 |
| `tmas` | `[E*2, 128]` int8 | 预构造的 TMA 描述符(Gate-Up + Down 各一份) |

## 4. Gate-Up GEMM:Routing-aware 路径

### 4.1 不再有显式 gather

普通 grouped GEMM 期望输入是已经按 expert 排好的 `[total, hidden]` 张量。HPC-Ops 走得更激进:

```
Gate-Up GEMM 的 X tensor 是原始 x[N, hidden],
TMA 描述符根据 topk_pos 直接索引到正确的行。
```

也就是说,**fuse_moe 的 Gate-Up GEMM 直接从原始 token 张量读,通过 routing 索引完成"软 gather"**。这省掉一次完整的 N*k*hidden 的内存拷贝。

源码:`src/fuse_moe/cp_async/fuse_moe.cu` 里的 `fuse_moe_gateup` kernel。

### 4.2 cp.async 路径,无 WS

`benchmark/fused_moe/README.md` 默认报告 fp8 路径,使用 cp.async 而非 WS。为什么?

低延迟场景下 batch 小,每个 expert 的 m_i 不大(比如 16-64),warp specialization 的 producer/consumer 拆分反而开销大(SMEM 占用、barrier 同步)。改用 cp.async + 让更多 CTA 驻留 SM,**整体并行度更高**。

### 4.3 PDL chain

Gate-Up GEMM 完成后,Activation + Down GEMM **几乎立即可以开始下一个 expert 的 token**(因为不同 expert 之间无数据依赖)。Programmatic Dependent Launch (PDL) 让下一个 kernel 提前启动,在 GPU 上无缝衔接。

## 5. Activation + Quant 融合

源码:`src/activation/activation.cu`

```python
# hpc/act.py
def masked_act_mul_and_quant(gate_up, scale, num_per_expert, output=None):
    """
    输入 gate_up: [N, 2C] bf16
    1. 分成 gate = gate_up[:, :C], up = gate_up[:, C:]
    2. silu(gate) * up
    3. 乘 scale (per-tensor)
    4. 量化到 fp8_e4m3
    5. 输出 [N, C] fp8
    """
```

为什么叫 "masked"?因为对每个 expert 输出的 token 数 `num_per_expert[e]` 不同,kernel 内部按 mask 跳过 padding 部分,避免对 expert 范围外的位置做不必要的写。

### 5.1 Blockwise quant 版本

`masked_act_mul_and_blockwise_quant` 输出额外一个 `output_scale: [N, C/128] fp32`:

```
对每 128 个元素一组算 scale:
    scale_block = max(abs(x_block)) / fp8_max
    fp8_block   = x_block / scale_block
```

这样后续 Down GEMM 可以走 blockwise GEMM 路径,精度更好。

## 6. Down GEMM 与 reduce

### 6.1 Down GEMM

输入是 Activation 后的 `[N*k, intermediate]`,按 expert 分组,权重 `[E, hidden, intermediate]`。同样是 grouped GEMM,kernel 在 `src/fuse_moe/cp_async/fuse_moe.cu` 的 `fuse_moe_down` 部分。

### 6.2 reduce:加权 scatter-add

最后一步:把每个 expert 输出的 token 按 `topk_scale` 加权,回填到原 token 位置。

```python
# hpc.reduce 等价于:
y = torch.zeros((N, hidden))
for i in range(N):
    for j in range(k):
        pos = topk_pos[i, j]
        y[i] += x_out[pos] * topk_scale[i, j]
y += shared_output  # 可选
```

源码:`src/fuse_moe/reduce.cu`。优化点:

- 每个 token 只读 k 次(k=8),不需要 atomic(因为每个 i 是独占的)。
- `hidden` 维度做 vector load(float4),warp 内并行。
- 把可选的 `shared_output` 在 kernel 内一并加上,省一次 kernel launch。

## 7. blockwise FP8 的"per-group 平均长度"参数

注意到 `fuse_moe_blockwise` 等接口里有一个 `num_seq_per_group_avg=32`:

```python
fuse_moe_blockwise(
    ...
    num_seq_per_group_avg=32,
)
```

含义:**估计每个 expert 平均收多少 token**。用于决定 grouped GEMM 的 kTileM(16/32/64):

- 小 batch 时 m_avg 小,选 kTileM=16,小 tile,SM 占用率高
- 大 batch 时 m_avg 大,选 kTileM=64,大 tile,Tensor Core 利用率高

这个参数是**性能 hint**而非正确性参数,设错了不会出错,只会慢。

## 8. 性能数字

来自 `benchmark/fused_moe/README.md`:

| 配置 | 对比 | 加速 |
|---|---|---|
| DeepSeek-V3, Hunyuan-V3, Qwen3-235B 在 TP=8 EP=1 | vLLM Triton, vLLM CUTLASS, SGLang | Up to 1.6x |
| 同上 在 TP=1 EP=8 | 同上 | Up to 1.5x |

复现:

```bash
python3 benchmark/fused_moe/benchmark_fuse_moe.py \
  --tp 8 --ep 1 \
  --providers hpcops vllm vllm_cutlass sglang \
  --csv fused_moe_tp8_ep1.csv
```

## 9. 与 vLLM / SGLang 的集成

vLLM / SGLang 的 MoE 实现可以**直接替换为 HPC-Ops**。`benchmark/fused_moe/backends/hpcops.py` 提供了适配代码,展示如何把它们的张量约定映射到 HPC-Ops。关键映射:

| vLLM 约定 | HPC-Ops 约定 |
|---|---|
| Activation FP8 量化用 `vllm._custom_ops.scaled_fp8_quant` | 也可以用 `hpc.scaled_fp8_quant`(产线上推荐) |
| topk 输出 `topk_ids, topk_weights` | `topk_ids: int32 [N,k]`, `topk_scale: fp32 [N,k]` |
| 共享 expert 输出 | `shared_output` 参数 |

## 10. 阅读源码路径

1. `hpc/fuse_moe.py`:Python 入口,看清三大 API 签名。
2. `src/fuse_moe/entry.cc`:C++ 入口与张量验证。
3. `src/fuse_moe/count_and_gather.cu`:routing 的实现,从这里学高效 SMEM atomic。
4. `src/fuse_moe/cp_async/fuse_moe.cu`:Gate-Up + Activation + Down 的真正流水线 kernel。
5. `src/fuse_moe/reduce.cu`:reduce / scatter-add 的实现,典型的 vector load + 累加。

---

下一篇:[05_fused_allreduce_rmsnorm.md](./05_fused_allreduce_rmsnorm.md) — 通信和计算的融合,Multicast 与 Lamport P2P。
