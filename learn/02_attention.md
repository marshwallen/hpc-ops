# 02 - Attention 算子家族

> 这是 HPC-Ops 里**最大**的算子家族,也是 LLM 推理里**最关键**的算子。建议结合 [00_background.md](./00_background.md) 中 Prefill/Decode、KV Cache、GQA 的解释一起阅读。

## 1. Attention 算子地图

HPC-Ops 在 `hpc/attention.py` 里导出了如下算子:

```
Prefill 类(处理一个 prompt 的所有 token):
  attention_prefill_bf16                       — 简单 prefill,无 KV cache,BF16
  attention_with_kvcache_prefill_bf16          — Paged KV cache,BF16
  attention_with_kvcache_prefill_fp8           — Paged KV cache,FP8 KV
  attention_with_kvcache_blocksparse_prefill_fp8  — 上面 + 块稀疏

Decode 类(一次生成 1+ 个 token):
  attention_decode_bf16                        — BF16 全精度
  attention_decode_fp8                         — FP8 KV + 量化 Q
  + 工具函数:
    get_attention_decode_task_workspace        — 分配动态调度 workspace
    assign_attention_decode_task                — 计算 task_map
    print_attention_decode_task                 — 调试用,打印调度结果
```

每一类还按**量化方案**(`QuantType` 枚举)展开:

```python
class QuantType(Enum):
    QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD = 0         # 最细粒度
    QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR = 1              # K/V 整张量一个 scale
    QPERTENSOR_KPERTENSOR_VPERTENSOR = 2                     # 全部 per-tensor
    QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD_QKHADAMARD = 3  # 带 QK Hadamard 旋转
```

## 2. 数据布局:理解算子签名的前提

所有 attention 算子都假设以下张量布局,**第一次看一定要对照源码的 docstring**:

### 2.1 Prefill 输入

```
q:               [total_seq, num_head_q, num_dim_qk]   bf16/fp8
                  └─ total_seq = sum(seqlens_q)
                  └─ 多条请求的 Q 直接 packed concat,变长

k, v:            [total_seq, num_head_kv, num_dim_qk]  ← prefill_bf16 不带 cache
                  或
kcache, vcache:  [num_blocks, block_size, num_head_kv, num_dim] ← 带 cache 的变体
                  └─ 这是 Paged KV Cache,见 00_background.md §1.3

cu_seqlens_q:    [num_batch + 1]   int32
                  └─ 累计前缀和,定位每条请求在 packed 张量里的起止
                     例如 cu_seqlens_q = [0, 8, 11, 23] 表示
                     batch 0 占 [0..8), batch 1 占 [8..11), batch 2 占 [11..23)

block_ids:       [num_batch, max_blocks] int32
                  └─ 每条请求用了哪些物理 cache 块

seqlens_kvcache: [num_batch] int32
                  └─ 每条请求在 KV cache 里有多少 token(prefill 前已存的)
```

### 2.2 Decode 输入

```
q:               [num_batch * num_seq_q, num_head_q, num_dim_qk]
                  └─ num_seq_q = mtp + 1 (multi-token-prediction)
                  └─ mtp=0 表示传统每步 1 个 token;mtp=1/2 是 speculative decoding

num_seq_kvcache: [num_batch]   int32  ← 类似 prefill 的 seqlens_kvcache
block_ids:       [num_batch, max_blocks]
```

### 2.3 NHD vs HND:同样数据,两种 stride

KV cache 的逻辑形状永远是 `[num_blocks, block_size, num_head_kv, num_dim]`,但**stride 可以不同**:

- **NHD 连续**(N = num_token, H = num_head, D = head_dim):标准 `contiguous` 张量。
- **HND 连续**:相当于 `tensor.transpose(1,2).contiguous().transpose(1,2)`,逻辑形状不变,但内存里 head 维度先于 token 维度。

HPC-Ops kernel **同时支持两者**,通过 `kcache.stride(1) / .stride(2)` 自动适应。这给了上层框架(vLLM/SGLang)更多选择自由。

## 3. Prefill 实现:Warp Specialization 范式

源码在 `src/attention/prefill/warp_spec_*.cu`。我们以 BF16 paged 版本为例。

### 3.1 高层思路

每个 CTA 负责 Q 的一个 tile(例如 128 个 token,1 个 head):

```
Output Y_tile = softmax(Q_tile @ K^T / sqrt(d)) @ V

       Q_tile: [TileM, head_dim]       已经在 SMEM
       K, V:   分块从 HBM 流入

外层循环 K/V 的 tile,内层做 FlashAttention 的在线 softmax:

  S = Q_tile @ K_tile^T               ← WGMMA fp16/bf16
  m_new = max(m_old, max(S))
  p = exp(S - m_new)
  l_new = exp(m_old - m_new) * l_old + sum(p)
  acc *= exp(m_old - m_new)
  acc += p @ V_tile                   ← WGMMA fp16/bf16
  m_old, l_old = m_new, l_new

最后:
  Y_tile = acc / l_old
```

### 3.2 Warp Specialization 拆分

```
一个 CTA 4 个 warp group(128*4=512 线程):

  WG 0: Producer    — 发起 TMA load 把 K_tile, V_tile 搬进 SMEM
                      load 完后,通过 mbarrier::arrive 通知 Consumer
  WG 1: Consumer    — wait barrier, 用 WGMMA 算 S = Q*K^T
                      wait barrier, 用 WGMMA 算 acc += P*V
  WG 2: Consumer    — 同上,但负责 Q 的另一半(增加并行度)
  WG 3: Producer    — 备用,或者参与 store
```

mbarrier 流水线让 Producer 提前 prefetch 1-2 个 stage,**充分隐藏访存**。

### 3.3 块稀疏的 epilogue 改造

`warp_spec_with_kvcache_blocksparse_fp8_dim128.cu` 在外循环里多检查一个 `block_mask[batch, head_q, tile_m, tile_kv]`:

```cpp
for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
    if (!block_mask[bid][hid][qtile][kv_tile]) {
        continue;  // 整个 tile 跳过,不发起 TMA load,不做 WGMMA
    }
    // 正常路径
}
```

效果:稀疏率 80% 的长上下文,实测 3 倍加速(`README.md` 中报告"Up to 3.16x")。**约束**:Q tile 的因果对角线那一块必须为 True,否则 softmax(全 -inf) → NaN。

## 4. Decode 实现:动态调度

### 4.1 为什么 decode 需要特殊调度

Decode 阶段每条请求**只产生 1-4 个 token**,但 K/V 维度可能从 256 长到 128K。如果按照"一 CTA 一请求"的方式调度:

- 短请求:CTA 几个 us 就结束,SM 闲置。
- 长请求:CTA 跑几十 ms,**形成 tail latency**。
- 混合 batch:整个 kernel 等最慢的那个 CTA。

### 4.2 静态 split-k(传统方案)

把每条请求的 KV 序列平均切成 K 段,每段交给一个独立 CTA,最后做"split combine"合并:

```
请求 A (KV=2048):
   ├─ CTA 0: KV[0..512)
   ├─ CTA 1: KV[512..1024)
   ├─ CTA 2: KV[1024..1536)
   └─ CTA 3: KV[1536..2048)
       └─ 4 个 partial 结果 → split_combine kernel → 最终输出

请求 B (KV=128):
   └─ CTA 4: 全部
```

**问题**:K 是编译期/静态决定的,不感知具体请求长度分布。请求 A 切成 4 份每份 512 时刚刚好,但如果 KV=256 切 4 份每份只有 64,反而过度细分。

### 4.3 HPC-Ops 的动态调度(SM90 新引入)

源码在 `src/attention/decode/sm90/dynamic/` 和 `src/attention/decode/assign_task.cu`。

**核心思路**:
1. 把所有请求的 KV 都按统一 tile 大小(`kTileN=64`)切片。
2. 用贪心 bin-packing 把这些 tile 分配到 N 个 CTA,让每个 CTA 的总工作量尽量均匀。
3. Kernel 读取生成好的"任务表"(task_map),按表执行。

**Task 数据结构**:

```cpp
// src/attention/decode/sched_task_info.h
struct alignas(16) TaskScheduleInfo {
    int ihead_kv;       // 哪个 KV head
    int ibatch;         // 哪条请求
    int ichunk;         // 这个请求里的第几个 tile
    int iseq_start;     // KV 起始位置
    int num_seqkv;      // 这个 tile 实际的 KV 长度
    int num_seqkvcache; // 总 KV 长度(用于 softmax 归一化)
    int num_tile_kv;    // 当前请求总共多少 tile
    int num_tile_full;  // 充满的 tile 数(用于 mask 边界)
    int is_casual_chunk;  // 是否处于因果对角线
    int pad[3];
};
```

**API 使用流程**(从 `hpc/attention.py` 看):

```python
# 1. 预分配 workspace(只跟最大 batch、最大 seqlen 有关)
workspace = hpc.get_attention_decode_task_workspace(
    max_num_batch=512, max_seqlen=131072, num_head_kv=8
)

# 2. 每个 decode step,根据当前 batch 的 num_seq_kvcache 算任务表
task_map = hpc.assign_attention_decode_task(
    num_seq_kvcache, workspace, num_head_kv=8, mtp=0, new_kv_included=False
)

# 3. 跑 attention,传入 task_map
out = hpc.attention_decode_fp8(
    q, kcache, vcache, block_ids, num_seq_kvcache,
    qscale, kscale, vscale, mtp=0,
    quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
    splitk=True,
    task_map=task_map,  # ← 关键
)
```

### 4.4 调度算法:贪心 bin-packing

`assign_attention_decode_task_sync`(`src/attention/decode/assign_task.cu`)在 GPU 上做:

```
输入: num_seq_kvcache[B], num_head_kv, num_seq_q
输出: 一个长度为 num_total_ctas 的"bin"序列,每 bin 装一组 tasks

伪代码:
  1. 枚举所有 (batch, head_kv) 对,把每对的 KV 序列按 tile 大小切片,
     得到 num_tasks 个 tile,每个 tile 工作量近似为 num_seqkv。
  2. 按 num_seqkv 降序排序 tasks(最大的先放)。
  3. 初始化 num_total_ctas 个空 bin,每个 bin 当前负载 = 0。
  4. 对每个 task,放入当前负载最小的 bin。
  5. 写出每个 bin 内的 task 序列到 task_map。
```

**实际实现**采用并行 GPU 算法做这个调度,延迟控制在几 μs。`assign_task.cu` 同时提供 CPU 版本(`assign_attention_decode_task_cpu_entry`)作为参考实现,用来在测试里验证 GPU 版本的正确性。

### 4.5 Combine Kernel

每个 CTA 写出一个 partial softmax 结果(包含 `m, l, acc`),最后一个 combine kernel 用 FlashAttention 的"在线 softmax 合并"公式合成最终输出:

```
for each output position:
    m_global = max over all CTAs' m_local
    l_global = sum over all CTAs' (l_local * exp(m_local - m_global))
    acc_global = sum over all CTAs' (acc_local * exp(m_local - m_global))
    output = acc_global / l_global
```

源码在 `src/attention/decode/splitk_combine_kernels.cuh`。

## 5. FP8 attention 的精度处理

FP8 范围只到 ±448,而 attention 的 softmax 前 logits 可以很大。需要 per-token / per-head 的 scale 来缩放。

### 5.1 几种 quant type 对比

| `QuantType` | Q scale | K scale | V scale | 适用 |
|---|---|---|---|---|
| `QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR` (=1) | `[B, H_q, max_seq_q_pad]` | `[1]` | `[1]` | 通用,scale 内存小 |
| `QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD` (=0) | `[B, H_q, max_seq_q_pad]` | `[num_blocks, block_size, H_kv, dim_scale]` | `[H_kv]` | 高精度,scale 内存大 |
| `QPERTENSOR_KPERTENSOR_VPERTENSOR` (=2) | `[1]` | `[1]` | `[1]` | 极简,适用于校准良好的模型 |

### 5.2 计算时的反量化

```
attention 等价于:
    Q_fp32 = Q_fp8 * qscale
    K_fp32 = K_fp8 * kscale
    P = softmax(Q_fp32 @ K_fp32^T / sqrt(d))
    V_fp32 = V_fp8 * vscale
    Y = P @ V_fp32
```

kernel 里把这些 scale 因子收敛到一个 `frob_scale = qscale * kscale * inv_sqrt(d)`,只在 softmax 前乘一次,避免重复缩放。

## 6. RoPE + KV 存储的融合

Attention 之前必经的步骤是:

1. 从 `qkv` projection 输出里拆 Q、K、V。
2. 对 Q、K 应用 RoPE 旋转位置编码。
3. (可选)对 Q、K 应用 RMSNorm。
4. 把新的 K、V 写入 paged KV cache 的指定槽位。
5. 输出 Q 供 attention 使用。

HPC-Ops 把这五步融合在 `rope_norm_store_kv` / `rope_norm_store_kv_fp8`(`hpc/rope.py`)里。详见 [07_normalization_rope_activation.md](./07_normalization_rope_activation.md)。

## 7. 测试方法

`tests/test_attention_decode_bf16.py` 是非常好的入门样本。它的结构:

```python
@pytest.mark.parametrize("num_batch", [1, 16, 200])
@pytest.mark.parametrize("num_seq_q", [1, 2, 3])
@pytest.mark.parametrize("max_seq_kv", [1024, 4096])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("kv_head_q_head", [(1, 4), (1, 8), (2, 16), (4, 32)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [True, False])
@pytest.mark.parametrize("splitk", [True, False])
def test_attention_decode_bf16(...):
    # 1. 随机生成 Q/K/V
    q = torch.randn(...) / math.sqrt(num_dim_qk)
    k = torch.randn(...)
    v = torch.randn(...)

    # 2. 构造 paged KV cache(随机洗牌 block_ids)
    kvcache = torch.randn(max_num_blocks, 2, block_size, num_head_kv, num_dim_qk)
    packed_block_ids = torch.randperm(max_num_blocks)[:total_blocks]
    ...

    # 3. 跑参考实现(PyTorch 暴力 attention)
    gt = ref_attn_with_paged_kvcache_func(q, k, v, kvcache, block_ids, ...)

    # 4. 跑 HPC-Ops
    my = hpc.attention_decode_bf16(q, kvcache[:, 0], kvcache[:, 1], block_ids, ...)

    # 5. 比对
    assert allclose(gt, my, atol=0.016)
```

跑测试:

```bash
# 一个具体参数
python3 -m pytest -v tests/test_attention_decode_bf16.py::test_attention_decode_bf16[True-True-128-\(1,4\)-64-1024-1-1]

# 全部参数(很多 case,慢)
python3 -m pytest -v tests/test_attention_decode_bf16.py
```

## 8. 性能数字(来自 README)

| 场景 | 对比基线 | 加速 |
|---|---|---|
| Attention BF16 prefill | FlashInfer / FA2 / FA3 / TensorRT-LLM | Up to 1.33x |
| Attention BF16 decode | 同上 | Up to 2.22x |
| Attention FP8 prefill | FlashInfer / FA3 / TensorRT-LLM | Up to 1.12x |
| Attention FP8 decode | 同上 | Up to 2.00x |
| Sparse Attention FP8 | MIT-BSA BF16, FA3-Dense FP8 | Up to 3.16x |
| Dynamic decode | 静态 split-k | Up to 2.88x |

复现命令在 `benchmark/attention_decode/README.md`。

## 9. 阅读源码的建议路径

1. **从 entry 开始**:`src/attention/entry.cc` 是 PyTorch 算子注册和形状检查,跟读它能理解每个张量的角色。
2. **追 decode dispatch**:`src/attention/decode/decode.cc` 极短,只是根据参数选具体实现。
3. **挑一个具体 kernel**:推荐先看 BF16 static split-k(`src/attention/decode/sm90/static/smallm_bf16_dim128_static.cu`),它最简单。然后看 FP8 dynamic 版本(`src/attention/decode/sm90/dynamic/smallm_fp8_*_dynamic.cu`)感受动态调度的代码组织。
4. **理解 CuTe 抽象**:CuTe(在 `3rd/cutlass/include/cute/`)是 NVIDIA 的张量层级抽象语言,所有 SM90 kernel 都用它写。建议读 [CUTLASS 官方 tutorial](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md) 后再回来看 kernel。

---

下一篇:[03_gemm_and_group_gemm.md](./03_gemm_and_group_gemm.md) — GEMM 与 Grouped GEMM,从 BF16×FP32 高精度合成讲起。
