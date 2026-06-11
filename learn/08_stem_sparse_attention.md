# 08 - Stem 块稀疏注意力

> 长上下文 prefill 时,KV 数量极大但大部分对当前 query 无关。**Stem 稀疏注意力**是一套"在线评分 + 选块"的稀疏化方案:用低维近似算每个 Q-block 对每个 KV-block 的重要度,只保留 top-K 的块给下游 FP8 attention。HPC-Ops 把整套 Stem 流程做成 4 个串联的 kernel,并提供端到端的 Python wrapper。

> Stem 与 [02_attention.md](./02_attention.md) 中的 `attention_with_kvcache_blocksparse_prefill_fp8` 配套:Stem 生成 `block_mask`,然后传给 block-sparse attention 算子。

## 1. Stem 稀疏注意力总览

四步管线:

```
                  q (FP8) + kvcache (FP8)
                          │
                          ▼
            ┌─ stem_oam_prep_paged_kv (KV 预处理) ─┐
            │                                      │
            └─ stem_oam_prep_varlen_q (Q 预处理) ──┤
                          │
                          ▼
                   stem_oam_gemm (OAM 评分)
                          │
                          ▼ block_logits [B, H_q, max_Qb, max_Kb]
                          │
                   stem_tpd (top-K + 政策选块)
                          │
                          ▼
                   mask [B, H_q, max_Qb, max_Kb] uint8
                          │
                          ▼
    attention_with_kvcache_blocksparse_prefill_fp8(mask=...)
```

`hpc.stem_paged_kv(...)` 是 Python 一站式入口,内部按顺序调用 4 个低层算子。

## 2. 各阶段细节

### 2.1 OAM 预处理

**OAM** = Overlap-Aware Matmul,把每个 block 的 token 加权求和到一个**低维表示**,这样后续的"block-vs-block"评分只需要做小矩阵乘。

```python
# KV 端
kflat, vbias = hpc.stem_oam_prep_paged_kv(
    kcache, vcache, kscale, vscale,
    kv_indices,           # [B, max_blocks_per_req]
    kv_seq_lens,          # [B]
    lambda_mag=0.3,       # V-bias 缩放系数
    stem_block_size=128,
    stem_stride=16,       # 下采样步幅
    quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
)
# kflat: [B, H_kv, max_Kb, stem_stride * dim_qk]   bf16
#        ↑ 每个 stem_block 用 stem_stride 个向量近似(反向 group order)
# vbias: [B, H_kv, max_Kb]   fp32
#        ↑ 每个 block 的 V 范数统计,做"重要度先验"

# Q 端
qflat = hpc.stem_oam_prep_varlen_q(
    q_fp8, qscale,
    q_seq_lens, cu_seqlens_q,
    stem_block_size=128, stem_stride=16,
)
# qflat: [B, H_q, max_Qb, stem_stride * dim_qk]    bf16
```

`stem_stride` 控制低维近似的粒度(默认 16 = 把 128 个 token 压成 16 个向量)。

### 2.2 OAM GEMM:block-level 评分

```python
block_logits = hpc.stem_oam_gemm(
    qflat, kflat, vbias,
    q_seq_lens, kv_seq_lens,
    stem_block_size=128, stem_stride=16,
    causal=True,
)
# block_logits: [B, H_q, max_Qb, max_Kb]    bf16
```

公式(单 head 角度):

```
block_logits[b, h, qb, kb] = FrobScale * (qflat[b, h, qb] · kflat[b, h_kv, kb])
                           + vbias[b, h_kv, kb]

其中:
  - 点积维度是 stem_stride * dim_qk = 16 * 128 = 2048
  - h_kv = h // group(GQA 共享)
  - causal=True 时,kb > qb 的位置直接设为 -inf
```

由于 `qflat` 和 `kflat` 都是 BF16,这个 GEMM 走 Tensor Core,速度极快;输出张量也比真实 attention 矩阵小很多(`max_Qb * max_Kb` vs `max_seq_q * max_seq_kv`)。

### 2.3 TPD: Top-K Policy Denoising

```python
mask = hpc.stem_tpd(
    block_logits,
    q_seq_lens, kv_seq_lens, num_prompt_tokens,
    block_size=128,
    alpha=1.0,                       # 每行预算线性衰减
    initial_blocks=4,                # 始终保留前 4 个 KV block(系统 prompt 区)
    window_size=4,                   # 始终保留对角线附近 4 个 block(局部窗口)
    k_block_num_rate_medium=0.2,     # 中等长度 prompt 的 top-K 比例
    k_block_num_bias_medium=30,
    k_block_num_rate_large=0.1,      # 长 prompt 的 top-K 比例
    k_block_num_bias_large=30,
)
# mask: [B, H_q, max_Qb, max_Kb]    uint8 (1=保留, 0=跳过)
```

#### TPD 内部逻辑

```
对每行 (b, h, qb):
  1. 算"基准 K 预算":
     prompt_kv_blocks = ceil(num_prompt_tokens[b] / 128)
     if prompt_kv_blocks < 56:
         K_base = prompt_kv_blocks   # 短:全保留
     elif prompt_kv_blocks < 160:
         K_base = round(prompt_kv_blocks * 0.2) + 30   # 中:20% + 30
     else:
         K_base = round(prompt_kv_blocks * 0.1) + 30   # 长:10% + 30

  2. 线性衰减:K_q = K_base * (1 - alpha * qb / max_Qb)  (alpha=1 时无衰减)

  3. 在 block_logits[b, h, qb, :] 上做 radix top-K,得到阈值 threshold
     标记 logits > threshold 的位置为 1

  4. 强制保留:
     mask[:, :, :, 0..initial_blocks-1] = 1                  # 初始系统 prompt
     mask[:, :, qb, qb-window_size+1..qb] = 1                # 对角窗口
     mask[:, :, qb, qb] = 1                                  # 因果对角(避免 NaN)

  5. 输出 uint8 mask
```

#### 设计要点

- **k_schedule 三段制**:短 prompt 全保留,中长 prompt 用不同比例。两段的 rate/bias 都暴露给用户调参。
- **chunked-prefill 不变性**:输入 `num_prompt_tokens` 始终是**完整 prompt 的总长度**(而非当前 chunk),保证不同 chunk 大小下 mask 一致,有利于 cache 命中和测试。
- **初始 + 窗口 + 对角线**强制保留:对应 attention 论文里的 "attention sink" 现象(前几个 token 异常重要)和 "local window"。

### 2.4 端到端 wrapper

```python
mask = hpc.stem_paged_kv(
    q_fp8, kcache, vcache,
    qscale, kscale, vscale,
    kv_indices, cu_seqlens_q, kv_seq_lens, num_prompt_tokens,
    lambda_mag=0.3, alpha=1.0,
    stem_block_size=128, stem_stride=16, causal=True,
    initial_blocks=4, window_size=4,
    k_block_num_rate_medium=0.2, k_block_num_bias_medium=30,
    k_block_num_rate_large=0.1, k_block_num_bias_large=30,
    quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
)
# 然后传给 block-sparse attention
out = hpc.attention_with_kvcache_blocksparse_prefill_fp8(
    q_fp8, kcache, vcache,
    qscale, kscale, vscale,
    cu_seqlens_q, block_ids, seqlens_kvcache,
    max_seqlens_q, quant_type=..., block_mask=mask,
)
```

## 3. 为什么需要这样设计

### 3.1 长上下文 + FP8 的双重挑战

```
普通 FP8 attention:计算 Q×K^T 全矩阵,然后 softmax,丢弃低分位置
              问题:Q×K^T 的内存与算力开销正比于 seq_len^2
                   对 64K 上下文 = 64K × 64K = 4G 元素,FP8 也要 4GB

Stem 稀疏:Q×K^T 只在 mask=1 的块上算,稀疏率 80% 时计算量降 5x
        但需要先生成 mask -> 这就是 Stem 的工作
```

### 3.2 评分阶段的开销小

OAM GEMM 的有效计算量:

```
完整 attention QK^T:  seq_q * seq_k * dim_qk
OAM GEMM:             num_Qb * num_Kb * (stem_stride * dim_qk)
                    = (seq_q / 128) * (seq_k / 128) * (16 * 128)
                    = (seq_q * seq_k) / 128
```

**评分开销是完整 QK 的 1/128**。这就是为什么"评分 + 稀疏 attention"整体能 3 倍加速:评分开销 1%,稀疏 attention 节省 80%。

### 3.3 FP8 量化贯穿整条管线

- `kcache`, `vcache`, `q` 都是 FP8 e4m3
- 各自带 dequant scale
- OAM 预处理时把 FP8 dequant 完转 BF16(为了 OAM GEMM 用 Tensor Core)
- OAM GEMM 用 BF16,输出 BF16
- TPD 在 BF16 上做 top-K(精度足够)

## 4. 数据布局细节

### 4.1 KV 端的 anti-diagonal 排序

`stem_oam_prep_paged_kv` 输出的 `kflat` 是**反向 group order**。原因是 anti-diagonal scoring 时,K 的最后一个 stride 对应"最近的 token",和 Q 的当前位置对应,匹配 OAM GEMM 的内积语义。

如果你想读懂 kernel 源码,看 `src/stem/stem_oam_prep_paged_kv_dim128.cu` 里的 group 反向 reshape 处。

### 4.2 V-bias 的物理意义

```
V_block 的能量 ≈ sum(|v|^2) / block_size
V_bias = lambda_mag * log(V_block_energy)  (大致)
```

把 V 的"信息量"作为先验加到 block_logits 上:**信息量大的 KV-block 即使 Q*K 分数一般也优先保留**。`lambda_mag=0.3` 是经验调参值。

### 4.3 quant_type 的不同 scale shape

| `quant_type` | kscale shape | vscale shape |
|---|---|---|
| `QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR` | `[1]` | `[1]` |
| `QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD` | `[num_blocks, scale_block_size, H_kv, dim_scale]` | `[H_kv]` |

第二种格式更精细,但 scale 张量大得多。kernel 内部分两条 dispatch path。

## 5. 性能数字(README)

| 配置 | 对比 | 加速 |
|---|---|---|
| FP8 sparse prefill, 不同稀疏率 + seq len | MIT-BSA BF16, FlashPrefill-BSA BF16, HPC-Dense FP8, FA3-Dense FP8 | Up to 3.16x |

在 64K 上下文、稀疏率 80% 的典型场景下,Stem + sparse attention 比 FA3 Dense FP8 快约 3 倍。

## 6. 与 dense attention 的互操作

`stem_paged_kv` 返回的 mask 直接喂给 `attention_with_kvcache_blocksparse_prefill_fp8`,二者形状对齐:

```
mask:   [B, H_q, max_Qb, max_Kb_in_mask]
attention block_mask: 完全一致
```

**注意**:`stem_block_size`(默认 128)必须等于 attention 的 `kTileN`(也是 128)。如果改 stem_block_size,需要同步改 attention 的 tile 大小,否则 mask 维度对不上。

## 7. 阅读源码路径

1. **Python 入口**:`hpc/stem.py` 全文 ~460 行,4 个底层算子 + 1 个 end-to-end wrapper。docstring 解释每个参数。
2. **stem kernels 共用工具**:`src/stem/stem_kernels.cuh`,通用 helper(group sum、anti-diag mapping)。
3. **OAM prep KV**:`src/stem/stem_oam_prep_paged_kv_dim128.cu`,从 paged FP8 cache 算 kflat 和 vbias。
4. **OAM prep Q**:`src/stem/stem_oam_prep_varlen_q_dim128.cu`。
5. **OAM GEMM**:`src/stem/stem_oam_gemm_dim128.cu`,带 causal mask epilogue 的 BF16 GEMM,典型 CuTe + WGMMA 写法。
6. **TPD**:`src/stem/stem_tpd.cu`,radix top-K 实现,加上 initial/window 强制保留。

## 8. 测试

`tests/test_stem_qpertoken_perhead_kvpertensor.py` 和 `tests/test_stem_qkpertoken_perhead_vperhead.py` 覆盖两种 quant_type。结构:

1. 用 dense PyTorch attention 算"基准分数"(完整 QK^T 后取 block-level top-K)
2. 跑 `hpc.stem_paged_kv` 得到 mask
3. 用 mask 跑 `hpc.attention_with_kvcache_blocksparse_prefill_fp8`
4. 与 dense FP8 attention 输出做 `allclose` 验证(允许稀疏化带来的误差)

---

下一篇:[09_communicator.md](./09_communicator.md) — 单节点 NVLink Multicast 通信器的内部实现,看 HPC-Ops 怎么用 CUDA Multicast API 把 NVSHMEM 替代掉。
