# 06 - Fused Sampler

> Decode 阶段输出 logits 后,需要做一连串"小操作":repetition penalty → temperature → softmax → top-k → top-p → 采样 → 更新 penalty mask。朴素实现是 7-10 个 PyTorch kernel,在 vocab=128K、batch=1-8 的低延迟场景下成为瓶颈。HPC-Ops Fused Sampler 把这条链子压成 1-2 个 CUDA kernel。

## 1. 采样链路速览

```
输入: logits [B, V]    其中 V = vocab_size (~128K)
                       B = effective batch (1-512)

完整 pipeline:
  1. repetition_penalty: 已出现过的 token 的 logit 缩小(避免重复)
  2. temperature scaling: logit /= T
  3. softmax (可选,在 top-k 前或后)
  4. top-k: 保留概率最高的 K 个候选
  5. softmax 在 top-k 上(如果上一步没做)
  6. top-p: 累积概率截断
  7. 采样: 从最终概率分布里采一个 token
  8. 更新 penalty mask: 记录采到的 token,下次 step 用
```

朴素 PyTorch 实现:每步一个 kernel,**10 次 vocab 维度的 IO**,每次 ~512KB。总开销可能 50+ μs,在 batch=1 时占 decode 总时间 30%+。

## 2. HPC-Ops 的两个采样算子

`hpc/sampler.py` 暴露一个统一入口 `fused_sampler`,内部自动分派两条路径:

### 2.1 完整路径:`fused_sampler`

```python
def fused_sampler(
    logits: Tensor,                                   # [B, V] fp32/bf16
    *,
    penalty_mask: Optional[Tensor] = None,            # [MAX_BS, V/8] uint8 bitmask
    slot_id: Optional[Tensor] = None,                 # [B] int32, mask 索引
    repetition_penalty: Union[Tensor, float] = 0.0,
    temperature: Union[Tensor, float] = 0.0,
    softmax_policy: SoftmaxPolicy = SoftmaxPolicy.NONE,
    topk: Union[Tensor, int] = 0,                     # [B] or scalar
    topp: Union[Tensor, float] = 0.0,
    max_topk: int = 32,                               # 编译期上界,32 或 64
    gumbel_noise: Optional[Tensor] = None,            # [B, V] fp32, 注入随机性
    draft_token_ids: Optional[Tensor] = None,         # [B] int64, speculative decode 用
    seed: int = 0,
) -> Tensor:
    """
    返回 token_ids: [B, 1] int32
    """
```

### 2.2 温度快速路径:`fused_sampler_temperature_sample`

当且仅当用户**只指定 temperature**(其他参数全是默认 0/None)时,wrapper 自动走轻量内核:

```python
# 自动检测条件
_fast_path = (
    penalty_mask is None and slot_id is None
    and repetition_penalty == 0.0 and topp == 0.0 and topk == 0
    and softmax_policy == NONE
    and temperature > 0  # 或者 tensor
)
```

快速路径跳过 cluster-cooperative top-k 机制,直接做:

```
score = logits / T + Gumbel(0)
token = argmax(score)
```

数学等价于 softmax 分布采样,但完全不计算 softmax,只需要一遍 vocab 扫描求 argmax。

## 3. `softmax_policy` 枚举

```python
class SoftmaxPolicy(IntEnum):
    NONE = 0          # 不做 softmax,top-k 直接操作 logit;Gumbel-max 采样
    BEFORE_TOPK = 1   # softmax 在 top-k 之前;top-k 操作概率
    AFTER_TOPK = 2    # 先 top-k 再 softmax(只对 K 个候选做)
```

**约束**:
- `topp > 0` 要求 `softmax_policy != NONE`(没有概率怎么定 p 阈值)
- `softmax_policy != NONE` 要求 `topp > 0`(否则 softmax 无意义,因为 Gumbel-max 不需要)

这两条约束在 C++ entry 里 `TORCH_CHECK` 强制。

## 4. Penalty Mask 的"bit-packing"设计

为什么用 `uint8 [MAX_BS, ceil(V/8)]` 而不是 `bool [MAX_BS, V]`?

```
V = 128K, MAX_BS = 512

bool 实现:  512 × 128K × 1 byte = 64 MB
bit 实现:   512 × 128K / 8 = 8 MB     ← 节省 8x
```

而且更新方式更原子化:

```cpp
// 把 slot s 的第 token_id 位置 1
atomicOr(&penalty_mask[s][token_id / 8], 1 << (token_id % 8));
```

每个 bit 一个 token,**8 个 token 共享一次原子操作**,在 atomic 拥塞少的场景下还更快。

## 5. 核心算法:Cluster-Cooperative Top-K

Top-K 在 V=128K 上是一个有挑战的问题。HPC-Ops 用 **SM90 cluster** + **本地堆/归并** 组合实现。

### 5.1 整体思路

```
Cluster = 4 个 CTA (在一个 GPC 内)
每个 CTA 负责 vocab 的 1/4

Phase 1 (各 CTA 本地):
  扫自己负责的 V/4 元素,维护一个小 K 的小顶堆(或并行 bitonic sort)
  得到 K 个 local-top 候选

Phase 2 (cluster 内归并):
  4 个 CTA 通过 cluster shared memory 把 4*K 个候选汇聚到主 CTA
  主 CTA 做最终 top-K 排序

Phase 3 (后续):
  对 top-K 做 softmax2 / topp / sample
```

`max_topk` 必须是 32 或 64,因为这是编译期决定的 SMEM 大小。

### 5.2 Reduction with softmax stats

为了避免之后还要重新扫一遍 vocab 算 softmax 的 max 和 sum,top-K 过程中**顺便累积**:

```
对 top-K 之外的元素:
  在 top-K 选拔时同时维护 max(logit) 和 sum(exp(logit - max))

输出 top-K 后,直接用这些统计量算 softmax,无需再扫一遍 vocab。
```

这是 FlashAttention 风格的"在线 softmax"思想在采样里的应用。

## 6. Gumbel-max 采样:无 softmax 的概率采样

### 6.1 数学原理

如果 `g_v ~ Gumbel(0)` 独立同分布,那么

```
argmax_v (logit_v + g_v) 
```

等价于从 `softmax(logit)` 采样。**完全不需要算 softmax!**

证明:Gumbel-max trick([Maddison et al. 2014](https://arxiv.org/abs/1411.0030))。

### 6.2 在 kernel 里如何获得 Gumbel(0)

两种方式:

1. **内部生成**:用 curand 在 kernel 里实时采 `u ~ Uniform(0,1)`,转换 `g = -log(-log(u))`。非确定性。
2. **外部注入**:用户传 `gumbel_noise: [B, V] fp32`,kernel 直接加。**bit-exact 可复现**,适合做单元测试。

```python
# 外部注入的标准生成方式
u = torch.rand(logits.shape, dtype=torch.float32, device=logits.device)
u = u.clamp_min_(1e-20)
gumbel_noise = -(-u.log()).log()
```

## 7. 完整 pipeline 在 kernel 中的顺序

`src/sampler/fused_sampler.cu` 内的 2-kernel 流水线:

```
Kernel 1 (per CTA cluster, V 维度分段处理):
  for v in my_segment:
    1. logit = logits[b, v]
    2. if (penalty_mask[slot_id[b]] bit v set):
         logit *= (logit > 0 ? 1/rp[b] : rp[b])           ← rep_penalty
    3. logit /= temperature[b]                             ← temperature
  
  (可选 softmax1 over full vocab)
  cluster-level top-K → 得到 K 个候选 + max + sum
  (可选 softmax2 over top-K)
  (可选 top-P 截断 with cum-sum)
  for each surviving candidate:
    score = value (or prob) + gumbel_noise[b, v]
    track local argmax
  cluster reduce → 全局 argmax 写出

Kernel 2 (per b, atomic update):
  token = token_ids[b]
  atomicOr(penalty_mask[slot_id[b]][token / 8], 1 << (token % 8))
```

## 8. 温度快速路径的具体实现

`src/sampler/fused_sampler_temperature.cu`:

```cpp
// 每个 CTA 负责一个 batch,V 维度内 thread-level 并行
__global__ void fused_sampler_temperature_kernel(
    int32_t* token_ids_out,
    const T* logits, int row_stride,
    const float* temperature_arr, float temperature_val,
    const float* gumbel_noise,
    const int64_t* draft_token_ids,
    int V, uint64_t rng_seed)
{
    int b = blockIdx.x;
    float T = temperature_arr ? temperature_arr[b] : temperature_val;
    int64_t drafted = draft_token_ids ? draft_token_ids[b] : -1;

    // 每个 thread 持有一个 (max_score, max_token) 对
    float best = -INFINITY;
    int best_token = -1;
    
    for (v = threadIdx.x; v < V; v += blockDim.x) {
        if (v == drafted) continue;  // 跳过 draft mask
        float l = (float)logits[b * row_stride + v];
        float score = l / T;
        score += gumbel_noise
            ? gumbel_noise[b * V + v]
            : curand_gumbel(rng_state);   // 内部生成
        if (score > best) {
            best = score;
            best_token = v;
        }
    }
    // CTA 内 reduce
    __shared__ float smem_scores[32];
    __shared__ int smem_tokens[32];
    warp_reduce_argmax(best, best_token);
    block_reduce_argmax(best, best_token, smem_scores, smem_tokens);
    if (threadIdx.x == 0) token_ids_out[b] = best_token;
}
```

- 每个 CTA 一行,简单清晰
- `draft_token_ids != -1` 的位置直接跳过(用于 speculative decoding 的 rejection sampling)

**约束**:此快速路径使用 per-device 共享 workspace(curand state),不能并发跨多 stream 调用同一 device。

## 9. 性能数字(来自 README)

| 配置 | 对比 | 加速 |
|---|---|---|
| BF16 logits, vocab=120832, batch 1-512 | vLLM-style PyTorch | Up to 8.5x |
| 同上 | FlashInfer | 视场景 |

复现:

```bash
python3 benchmark/sampler/benchmark_sampler.py \
  --timing nsys \
  --output-dir sampler_nsys
```

为什么这么快?
- 朴素 PyTorch 7 个 kernel × ~3μs launch = 21 μs **纯 overhead**
- HPC-Ops 1-2 个 kernel,总 launch overhead ~6 μs
- Vocab IO 从 7 次 → 1 次,IO 时间近 7x 降低

## 10. 测试方法:bit-exact 验证

`tests/test_sampler.py` 提供了纯 PyTorch 参考实现 `ref_fused_sampler`,通过**外部注入 Gumbel(0)**实现 bit-exact 对比:

```python
def test_fused_sampler_bit_exact():
    logits = torch.randn(B, V, dtype=torch.bfloat16, device="cuda")
    gumbel = _gumbel0_like(logits)
    
    ref_token, ref_mask = ref_fused_sampler(
        logits, temperature=0.7, topk=32, topp=0.9,
        softmax_policy=SoftmaxPolicy.AFTER_TOPK,
        gumbel_noise=gumbel,
    )
    my_token = hpc.fused_sampler(
        logits, temperature=0.7, topk=32, topp=0.9,
        softmax_policy=SoftmaxPolicy.AFTER_TOPK,
        gumbel_noise=gumbel,
    )
    assert torch.equal(ref_token, my_token)
```

`ref_fused_sampler` 严格模拟 kernel 的数值流程(包括 `effK = max_topk` 的隐式约束),这样 bit-exact 才能站得住。

## 11. 边界情况说明

### 11.1 topk=0 不是"全 vocab 采样"

文档里特别强调:

> ``topk=0`` (or unset) does **NOT** widen sampling to the full vocab: it just means "do not tighten below ``max_topk``", so sampling still happens within the top-``max_topk`` (32/64) candidates.

也就是说**始终是 top-`max_topk`(32/64) bounded sampling**,低概率长尾 token 不会被采到。这与 vLLM 的"完整 vocab Gumbel-max 采样"语义不同,**对齐时要注意**。

### 11.2 temperature ≤ 0 不允许

`fused_sampler_temperature_sample` 在 kernel 内部不做 `t > 0` 检查(为了性能),所以 entry 端 `TORCH_CHECK(min(t) > 0)` 严格守门。

### 11.3 stream 并发不安全

```
The temperature-only fast path (and fused_sampler_temperature) uses
a per-device shared workspace. Do NOT invoke this sampler concurrently
on multiple streams of the same device — use one stream per device or
serialize the calls.
```

如果你的 server 用多 stream,请在 sampler 调用前手动同步。

## 12. 阅读源码路径

1. **Python wrapper**:`hpc/sampler.py` 全文 ~180 行,看完就懂用法 + 快速路径分派。
2. **C++ entry**:`src/sampler/entry.cc`,所有验证逻辑(约束、shape 检查)集中在这里。
3. **Sampler header**:`src/sampler/sampler.h`,两个 launcher 函数签名,顶部注释精确描述算法语义。
4. **Temperature fast-path**:`src/sampler/fused_sampler_temperature.cu`,最简单的入门样本。
5. **完整 sampler**:`src/sampler/fused_sampler.cu`,cluster-cooperative top-K + softmax + topp + Gumbel-max。
6. **RNG**:`src/sampler/sampler_rng.cuh`,curand 状态管理。

---

下一篇:[07_normalization_rope_activation.md](./07_normalization_rope_activation.md) — 几个"小算子"的融合策略。
