# 03 - GEMM 与 Grouped GEMM

> 矩阵乘是深度学习的"原子"。HPC-Ops 提供两类自研 GEMM:
>
> 1. **BF16×FP32 双精度合成 GEMM**:面向 MoE router 和 sparse attention 的状态压缩等"精度敏感小矩阵"。
> 2. **Grouped GEMM**:面向 MoE 专家并行,一次 launch 完成 N 个不同 shape 的 GEMM。

## 1. 为什么不直接用 cuBLAS / CUTLASS?

`cuBLAS` 和 `CUTLASS` 的通用 GEMM 已经是 SOTA。HPC-Ops 在这两个场景下需要做"针对场景的特化":

| 场景 | cuBLAS/CUTLASS 不足 | HPC-Ops 的做法 |
|---|---|---|
| MoE Router GEMM(N=192, K=4096,FP32 权重)| FP32 没有 Tensor Core 加速,只能走 CUDA Core,慢;TF32 精度不够 | 把 FP32 权重拆成两个 BF16(high+low), 用 2 次 BF16 Tensor Core GEMM 合成 FP32 |
| Grouped MoE GEMM(N 个不同长度的 sub-GEMM 一起跑)| 通用 grouped GEMM 启动一次 launch 一次,小 sub-GEMM 浪费 SM | 自研 grouped GEMM,共享 launch,统一调度 tiles 给所有 sub-GEMM |

## 2. BF16 × FP32 GEMM

### 2.1 算法核心:双 BF16 合成 FP32

观察:FP32 权重 `w_fp32` 可以无损地拆成

```
w_fp32 ≈ w_high + scale * w_low

其中:
    w_high = bfloat16(w_fp32)              # 截断到 BF16
    w_low  = bfloat16((w_fp32 - w_high.float()) / scale)
    scale  = 1.0 / 256.0                   # 固定缩放
```

`scale=1/256` 的选择是让 `w_low` 落在 BF16 的有效精度区间(BF16 尾数 7 bit ≈ 1/256)。

于是

```
y = x @ w_fp32^T
  ≈ x @ w_high^T + scale * (x @ w_low^T)

= GEMM_bf16(x, w_high) + scale * GEMM_bf16(x, w_low)
```

变成**两次 BF16 Tensor Core GEMM**,精度接近 FP32(累加用 FP32 register)。

### 2.2 关键优化:融合在一个 kernel

如果分别 launch 两个 GEMM 再做加法,会有:
- 2 次 kernel launch overhead
- 中间结果写回 HBM,~1MB 传输

HPC-Ops 在一个 kernel 里:

```
1. TMA load x_tile, w_high_tile 进 SMEM(stage 0)
2. WGMMA: acc_high += x_tile @ w_high_tile     ← FP32 累加器
3. TMA load x_tile, w_low_tile 进 SMEM(stage 1)
4. WGMMA: acc_low  += x_tile @ w_low_tile      ← FP32 累加器
5. 结果:y_tile = acc_high + scale * acc_low
6. (可选)bf16 转换、TMA store
```

注意:`x_tile` 只 load 一次,被两次 GEMM 共用。

### 2.3 Split-K + 跨 CTA 同步

当 `K` 很大(4096)而 `M, N` 都很小(M=16, N=192)时,普通 GEMM 一个 CTA 占满 K 维度,SM 利用率低。**Split-K** 把 K 拆成 S 段,各 CTA 算一段然后求和。

但 split-k 需要"最后一个完成的 CTA 做 reduce"。HPC-Ops 通过 `split_flag` 张量(`atomicAdd` 计数器)实现:

```cpp
// 伪代码
__global__ void gemm_bf16xfp32_splitk_kernel(...) {
    // 1. 计算本 CTA 负责的那段 K
    acc = compute_partial(x_slice, w_slice);

    // 2. 累加到全局 splitk_y 的对应位置(atomic)
    atomicAdd(splitk_y + tile_offset, acc);

    // 3. 用 split_flag 的 atomicInc 看自己是不是最后一个
    int finish_count = atomicAdd(split_flag + tile_id, 1);
    if (finish_count == splitk - 1) {
        // 我是最后一个,负责把 splitk_y 写到 y(可选量化)
        y[tile_offset] = splitk_y[tile_offset];
    }
}
```

### 2.4 Python API

```python
import hpc

# 准备权重(假设 w_fp32 是 [n, k] 的 FP32)
scale = 1.0 / 256.0
w_high = w_fp32.to(torch.bfloat16)
w_low = ((w_fp32 - w_high.float()) / scale).to(torch.bfloat16)
scale_t = torch.tensor([scale], dtype=torch.float32, device="cuda")

# 可选:预分配 split-k 标志
split_flag = hpc.get_gemm_bf16xfp32_workspace(max_weight_hidden_size=192)

# 调用
y = hpc.gemm_bf16xfp32(
    x,                  # [m, k] bf16
    w_high,             # [n, k] bf16
    w_low,              # [n, k] bf16
    scale_t,            # 标量
    use_fp32_output=False,  # 输出 bf16 (False) 或 fp32 (True)
    use_splitk=True,
    split_flag=split_flag,
)
# y: [m, n] bf16
```

### 2.5 性能数字

来自 `benchmark/route_gemm/`:

| Baseline | M sweep | HPC-Ops 加速 |
|---|---|---|
| cuBLAS FP32 | M=2..4096, N=192, K=4096 | Up to 3.22x |
| cuBLAS TF32 | 同上 | Up to ~1.5x |

精度对照 FP64 参考几乎无损(`max_abs_error` 接近 FP32 cuBLAS 水平)。

## 3. Grouped GEMM

### 3.1 什么是 Grouped GEMM

考虑 MoE:输入 token 分发到 G 个 expert,每个 expert 收到不同数量的 token:

```
Expert 0: 收到 m_0 = 12 个 token   ─┐
Expert 1: 收到 m_1 = 8 个 token     ├─ 每个 expert 是 [m_i, k] @ [n, k]^T → [m_i, n]
Expert 2: 收到 m_2 = 25 个 token    │
Expert 3: 收到 m_3 = 5 个 token    ─┘
```

如果每个 expert 都 launch 一个 GEMM,有 G 次 launch overhead;且 m_i 很小,单个 CTA 都装不满 SM。

**Grouped GEMM**:一次 launch,kernel 内部同时调度 G 个 sub-GEMM 的所有 tiles。

### 3.2 关键数据结构

```python
# Python 入参(见 hpc/group_gemm.py)
x:           [total_seq, hidden_size]        fp8 / bf16
             └─ total_seq = sum(m_i)
weight:      [num_group, output_dim, hidden_size]  fp8 / bf16
             └─ 每个 expert 一份 weight
seqlens:     [num_group]                     int32
             └─ m_0, m_1, ..., m_{G-1}
cu_seqlens:  [num_group + 1]                 int32
             └─ 0, m_0, m_0+m_1, ..., total_seq
y_scale:     [num_group]                     float32  (per-tensor 模式)
             或
x_scale:     [hidden_size//128, total_seq_pad]    float32  (blockwise 模式)
w_scale:     [num_group, output_dim//128, ...]    float32
```

### 3.3 内部:Tile 调度

源码在 `src/group_gemm/`。核心思想:

```
1. 在 host 端(其实是预 kernel)计算每个 expert 的 tile 数:
       tiles[g] = ceil(m_g / kTileM) * ceil(n / kTileN)
   累加得到 cu_tiles。

2. 主 kernel 的每个 CTA 拿到一个 "global tile id":
       global_tile_id = blockIdx.x
   通过 cu_tiles 二分查找,定位是哪个 expert 的第几个 tile:
       expert_id = upper_bound(cu_tiles, global_tile_id) - 1
       local_tile_id = global_tile_id - cu_tiles[expert_id]

3. 用 expert_id 索引到正确的 weight、scale,做 WGMMA。
```

这种调度让所有 expert 的 tiles 充分填满 SM,**变长 batch 自然均衡**。

### 3.4 TMA 描述符:为每个 expert 预构造

每个 expert 的 weight 是不同的张量(逻辑上 `weight[g]`),TMA load 需要不同的描述符。

HPC-Ops 在算子入口预构造好 `tmas: [num_expert * 2, 128]` 张量(每个描述符 128 字节,2 个分别给 weight 的 LOAD 和 output 的 STORE),kernel 直接根据 `expert_id` 取描述符。

### 3.5 Blockwise FP8:更细粒度量化

Per-tensor FP8 对小 expert 的累积误差大。**Blockwise FP8** 把 weight 沿 K 维度每 128 一个 block,各 block 独立 scale:

```
w_scale 形状: [num_group, output_dim // 128, (k // 128 + 3) // 4 * 4]
            └─ 每 128×128 个 fp8 weight 一个 fp32 scale
            └─ 最后一维 pad 到 4 的倍数(为了对齐 TMA load)

x_scale 形状: [hidden_size // 128, total_seq_pad]
            └─ x 同样按 K 维度分块 scale
```

kernel 在 K 方向的累加循环里乘 scale。`reformat_x_scale` 算子负责把 x_scale 从 deepep 标准格式重排为 HPC-Ops kernel 期望的紧凑格式。

### 3.6 cp.async 路径:为什么有 `group_gemm/cp_async/`

`group_gemm/group_gemm_pertensor_fp8.cu` 走 WS 路径(高吞吐)。
`group_gemm/cp_async/group_gemm_fp8.cu` 走 cp.async 路径(低延迟)。

差别:

| | WS 路径 | cp.async 路径 |
|---|---|---|
| Pipeline 实现 | CTA 内 producer/consumer warp 分工 | CTA 单一职责,流水线在多 CTA 间靠硬件调度 |
| CTA 占用 SM 资源 | 高(SMEM 占用大) | 低(SMEM 占用小) |
| SM 上 CTA 残留数 | 1-2 | 4-8 |
| 适用 batch | 大 | 小 |

低 batch 时,cp.async 路径的 CTA 多,**SM 利用率高**;且 cp.async 提交后即可继续,**延迟隐藏**靠硬件调度跨 CTA 实现,而非单 CTA 内的软件 pipeline。

### 3.7 性能数字(来自 README)

| Baseline | 场景 | 加速 |
|---|---|---|
| DeepGEMM | Expert grouped matmul prefill | Up to 1.1x |
| DeepGEMM | Expert grouped matmul decode | Up to 1.88x |

decode 阶段 batch 很小,cp.async 路径优势明显。

## 4. CuTe + WGMMA 的代码模式

HPC-Ops 的所有 SM90 GEMM 都使用 CUTLASS 的 CuTe 张量层级抽象。这里给一段最简化的代码骨架,帮助你看懂 `src/group_gemm/group_gemm_pertensor_fp8.cu`:

```cpp
// 编译期参数
using Config = GroupGEMMFp8Config<
    cutlass::float_e4m3_t,    // Tin
    cutlass::bfloat16_t,      // Tout
    /*kTileM=*/64,
    /*kTileN=*/128,
    /*kTileK=*/64,
    /*kStage=*/4              // 流水线深度
>;

// 1. 选 swizzle layout(把内存模式编排为 GMMA 期望的形状)
using SLayoutXAtom = Layout_K_SW128_Atom<float_e4m3_t>;   // 由 slayout_selector 选

// 2. 把 atom 扩展到完整的 [TileM, TileK, kStage] 形状
using SLayoutX = decltype(tile_to_shape(
    SLayoutXAtom{}, make_shape(Int<64>{}, Int<64>{}, Int<4>{})
));

// 3. 用 make_tma_copy 在 host 端构造 TMA 描述符
auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, x_tensor, take<0,2>(SLayoutX{}));
auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, w_tensor, take<0,2>(SLayoutW{}));
auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y_tensor, CopyBoxY{});

// 4. WGMMA tiled instruction
using TiledMma = decltype(make_tiled_mma(
    SM90_64x128x32_F32E4M3E4M3_SS_TN<>{},        // 单条 wgmma 指令
    Layout<Shape<_2, _1, _1>>{}                  // warpgroup 排布:2x1=2 个 warpgroup 一起
));

// 5. kernel 内伪代码
__global__ void kernel(TmaX tma_x, TmaW tma_w, TmaY tma_y, ...) {
    extern __shared__ char smem_raw[];
    auto sX = make_tensor(make_smem_ptr<Tin>(smem_raw),       SLayoutX{});
    auto sW = make_tensor(make_smem_ptr<Tin>(smem_raw + ...), SLayoutW{});

    // Producer / Consumer 分工
    if (warpgroup_idx == 0) {
        // Producer: TMA load 进 sX, sW
        for (k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
            int stage = k_tile % kStage;
            tma_x_load(stage);
            tma_w_load(stage);
            mbarrier_arrive(produce_bar[stage]);
        }
    } else {
        // Consumer: WGMMA
        FragmentAcc acc;
        for (k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
            int stage = k_tile % kStage;
            mbarrier_wait(produce_bar[stage]);
            warpgroup_arrive();
            cute::gemm(TiledMma{}, sX(_, _, stage), sW(_, _, stage), acc);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            mbarrier_arrive(consume_bar[stage]);
        }
        // store via TMA
        ...
    }
}
```

如果你看不懂 `make_tensor`、`make_tma_copy` 这些 CuTe API,强烈建议先读 CUTLASS 官方教程:
- https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md

## 5. 阅读源码路径

1. **从头文件开始**:`src/group_gemm/group_gemm.h` 列出所有 async launcher。
2. **配置类**:`src/group_gemm/config.h` 是核心,所有 tile 形状、swizzle、warpgroup 排布都在这里参数化。
3. **per-tensor 实现**:`src/group_gemm/group_gemm_pertensor_fp8.cu` 是 WS 路径的入门样本。
4. **blockwise 实现**:`src/group_gemm/group_gemm_blockwise_fp8.cu` 加上 K 维度的 scale 处理。
5. **cp.async 路径**:`src/group_gemm/cp_async/` 是另一套(低延迟)实现,适合 decode。
6. **bf16×fp32**:`src/gemm/sm90/gemm_bf16xfp32.cu` 是双 BF16 合成 FP32 的具体 kernel。

---

下一篇:[04_fused_moe.md](./04_fused_moe.md) — Fused MoE,如何把 router + GEMM + activation + GEMM + reduce 融成一条流水线。
