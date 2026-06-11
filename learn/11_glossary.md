# 11 - 术语表

> 按主题排序,查不到的回到对应章节;每个条目尽量给"一句话直观解释 + HPC-Ops 里在哪里出现"。

## A. 推理流程相关

**Prefill(预填充)**
处理 prompt 的所有 token,一次性算出 attention 上下文。**计算密集**。HPC-Ops:`attention_prefill_bf16` / `attention_with_kvcache_prefill_*`。

**Decode(自回归解码)**
逐 token 输出,每次只新增 1-N 个 token。**访存密集**。HPC-Ops:`attention_decode_bf16` / `attention_decode_fp8`。

**MTP(Multi-Token Prediction)**
一次 forward 输出多个 token(2-4),配合 speculative decoding 使用。HPC-Ops 的 decode 算子里的 `mtp` 参数。

**Speculative Decoding(投机解码)**
用小模型先猜几个 token,大模型一次验证,加速 decode。HPC-Ops 通过 `draft_token_ids` 参数支持 rejection sampling。

**KV Cache**
缓存历史 token 的 K 和 V 张量,避免每次重算。

**Paged KV Cache**
把 KV cache 切成固定大小块(block_size=32/64),每条请求维护"块 ID 表"。vLLM 首创,HPC-Ops 全面采用:`kcache: [num_blocks, block_size, num_kv_heads, dim]` + `block_ids: [num_batch, max_blocks]`。

**Block-Sparse Attention(块稀疏注意力)**
attention 的 KV 维度按块跳过部分计算。HPC-Ops 通过 `block_mask` 参数支持。

**TP(Tensor Parallel,张量并行)**
把 weight 横向切到多 GPU,中间结果需要 AllReduce 合并。HPC-Ops 的 `fuse_allreduce_rmsnorm_*` 服务 TP。

**EP(Expert Parallel,专家并行)**
MoE 的 expert 按维度分布到多 GPU,routing 时按 expert id 分发。HPC-Ops 的 fuse_moe 通过 `rank_ep` 和 `num_expert_total` 参数支持。

## B. 模型架构相关

**MHA(Multi-Head Attention)**
标准多头注意力,Q 和 K 头数相同。

**GQA(Grouped Query Attention)**
每 G 个 Q 头共享一对 KV 头。KV cache 缩小 G 倍。HPC-Ops:`num_head_q / num_head_kv` ∈ {4, 8}。

**MQA(Multi-Query Attention)**
GQA 的极端情况:`num_head_kv = 1`。

**MoE(Mixture of Experts)**
模型由多个 expert(独立 FFN)组成,router 给每个 token 选 top-k 个 expert。

**RoPE(Rotary Positional Embedding)**
通过旋转把位置信息编码进 Q、K。

**RMSNorm**
LayerNorm 的简化版,只做 RMS 缩放,不减均值。

**SwiGLU**
SiLU(gate) × up,FFN 的常用激活组合。

## C. 数据类型 / 量化

**FP32 / TF32**
32 位浮点。FP32 全精度;TF32 是 Ampere Tensor Core 用的 19 位假 FP32。

**BF16**
16 位浮点,8 位指数 + 7 位尾数。范围同 FP32,精度低。**LLM 训练 + 推理的默认精度**。

**FP16**
16 位浮点,5 位指数 + 10 位尾数。范围 ±65504,容易溢出。

**FP8 E4M3 / E5M2**
8 位浮点。E4M3:范围 ±448,精度高(主用)。E5M2:范围 ±57344,精度低(累加器用)。HPC-Ops 的 FP8 算子用 `torch.float8_e4m3fn`。

**Per-tensor scale**
整个张量一个 fp32 scale。最简,最快。

**Per-token scale**
每个 token 一个 scale。中等粒度。

**Per-head scale**
每个 attention head 一个 scale。

**Blockwise scale**
每 128(或其他)元素一个 scale。最精细,常用于 fp8 GEMM。

**QuantType**(HPC-Ops 枚举)
```
QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD = 0   # 最精细
QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR = 1        # 通用
QPERTENSOR_KPERTENSOR_VPERTENSOR = 2               # 极简
```

## D. GPU 硬件相关

**SM(Streaming Multiprocessor)**
GPU 的"核",H100 有 132 个 SM。

**CTA(Cooperative Thread Array)**
也叫 Thread Block,一个 grid 由多个 CTA 组成。一个 CTA 绑定到一个 SM。

**Warp**
32 个线程的最小调度单元。SIMT。

**Warp Group / Warpgroup(WG)**
4 个 warp(128 线程),SM90 的 WGMMA 操作单位。

**Cluster(线程块簇)**
SM90 新增,一个 Cluster 内多个 CTA 可共享 DSMEM 和 cluster barrier。

**SM90**
Hopper 架构(H100/H20/H200),也叫 Compute Capability 9.0。HPC-Ops 主要 target。

**SM90a**
SM90 的扩展指令集(WGMMA、TMA 等),编译时需要 `-arch=sm_90a` 而非 `sm_90`。

**Tensor Core**
专用矩阵乘加单元,Volta 起加入。一次做 16×16 大小矩阵乘加。

**CUDA Core**
传统 SIMT 标量计算单元。

**HBM(High Bandwidth Memory)**
GPU 显存,H100 是 HBM3(3 TB/s)。

**SMEM / Shared Memory**
SM 内的 fast scratchpad,~228 KB/SM(H100)。

**Register**
线程私有寄存器,~64K regs/SM。

**L1 / L2 Cache**
GPU 缓存层级。L1 与 SMEM 共享物理资源,L2 是全 GPU 共享(50 MB/H100)。

## E. CUDA 编程范式

**WGMMA(Warpgroup Matrix Multiply Accumulate)**
SM90 新指令,4 个 warp 一拍做大矩阵乘加。**异步**,可与其他操作重叠。

**TMA(Tensor Memory Accelerator)**
SM90 硬件单元,专门做显存↔SMEM 的张量级搬运,自动处理 stride、swizzle、边界。

**TMA Descriptor**
128 字节的描述符,告诉 TMA "搬什么形状的张量"。HPC-Ops 在 host 端构造,放在 `tmas` 张量里传给 kernel。

**cp.async**
SM80 引入的异步全局内存→SMEM 拷贝指令。SM90 上仍可用,比 TMA 灵活但粒度小。

**Warp Specialization(WS)**
CTA 内 warp 分工:Producer 负责 TMA load,Consumer 负责 WGMMA 计算,通过 mbarrier 通信。隐藏访存延迟。

**mbarrier**
SMEM 上的硬件同步原语,SM90 用于 WGMMA 与 TMA 的 producer/consumer 通信。

**Cooperative Groups(cg::cluster_group, cg::grid_group)**
CUDA 提供的协作同步 API。HPC-Ops 的 allreduce 大量使用。

**PDL(Programmatic Dependent Launch)**
SM90 新特性,允许两个 kernel 重叠 launch,前者尾部和后者头部并行执行,**消除 kernel 间气泡**。HPC-Ops 的 fuse_moe / allreduce_low_latency 使用。

**CGA(Cluster Grid Array)**
即 cluster + grid 抽象。SM90 的 cluster 是 grid 的 sub-level。

**Multimem / Multicast(NVLink 多播)**
NVLink 4 + NVSwitch 提供,一次写到 multicast address,多个 GPU 同时收到。HPC-Ops 的 allreduce_high_throughput 核心机制。

**Multimem Reduction**
multimem 操作的变种,load 时硬件做 sum/min/max,适合 AllReduce。

**Lamport Protocol**
不需要显式 flag 的 P2P 一致性协议:通过"是否为 negative-zero"判断数据是否就绪。HPC-Ops 的 allreduce_low_latency 用。

**curand**
CUDA 随机数库,HPC-Ops sampler 内部生成 Gumbel noise 时使用。

## F. CUTLASS / CuTe 抽象

**CUTLASS**
NVIDIA 开源的 CUDA C++ template 库,实现高性能 GEMM。HPC-Ops 把 `cutlass/include/cute/` 直接 vendor 进 `3rd/`。

**CuTe**
CUTLASS 3.x 的张量层级抽象,描述 layout 和 tile。HPC-Ops 所有 SM90 kernel 用 CuTe 写。

**Layout / Shape / Stride / Swizzle**
CuTe 的核心概念。Layout 描述张量逻辑布局和物理 stride;Swizzle 是为了 Tensor Core 高效访问而对 SMEM 地址做的位 XOR 变换。

**Tiled MMA**
一组 WGMMA 指令的组合(比如 2 warpgroup × 1 = 2 wg 同时跑)。

**TMA Atom / Layout Atom**
CuTe 用 "atom" 描述最小可重用的硬件操作单元。

## G. 其他 LLM 推理术语

**LSE(Log-Sum-Exp)**
FlashAttention 风格 attention 中用来合并多个 split-k 结果的归一化量。

**Split-K**
GEMM 的 K 维度切分,多 CTA 各算一段然后求和。HPC-Ops 的 decode attention 用。

**Combine Kernel**
split-k attention 的最后一步,合并多个 partial 结果。

**Online Softmax**
FlashAttention 的核心 trick:边算 QK 边更新 max 和 sum,一遍过。

**Penalty Mask**
记录哪些 token 已经被 sample 过,用于 repetition penalty。HPC-Ops:`[MAX_BS, ceil(V/8)] uint8` bit-packed。

**Gumbel-Max Trick**
`argmax(logit + Gumbel(0))` 等价于从 softmax 采样,**无需 softmax**。HPC-Ops sampler 用此机制。

**Top-K / Top-P(Nucleus)Sampling**
top-K:保留概率最高的 K 个。top-P:保留累计概率到 P 的最少 token。HPC-Ops sampler 全支持。

**Stem(Stem 稀疏)**
HPC-Ops 自研的块稀疏 attention 评分方案。OAM 预处理 → OAM GEMM → TPD 选块。

**OAM(Overlap-Aware Matmul)**
Stem 中的 block-level 评分 GEMM,把每个 block 用低维向量近似后做内积。

**TPD(Top-k Policy Denoising)**
Stem 的选块策略:top-K + 初始 + 窗口 + 因果对角线强制保留。

## H. 数据布局缩写

**NHD layout**
KV cache 维度顺序 `[num_blocks, block_size(N), num_heads(H), head_dim(D)]`,N 在 H 前面。标准 contiguous。

**HND layout**
同形状但 H 在 N 前面。通过 stride 重排表达,HPC-Ops 同时支持两者。

**LDx**(Leading Dimension)
矩阵的 stride。`ldQ = q.stride(0)` 等。出现在 HPC-Ops 的内部 launcher 参数里。

## I. 工具链

**NCCL**
NVIDIA Collective Communications Library。HPC-Ops 的 baseline 之一。

**NVSHMEM**
NVIDIA Shared Memory(NVLink/InfiniBand)PGAS 通信库。HPC-Ops **不依赖** NVSHMEM。

**FlashAttention(FA1/FA2/FA3)**
NVIDIA + 高校开源的 attention 实现。HPC-Ops attention 的对比 baseline。

**FlashInfer**
开源 LLM serving kernel 库。HPC-Ops 的 baseline 之一。

**TensorRT-LLM**
NVIDIA 闭源(部分开源)的 LLM 推理库。HPC-Ops 的 baseline 之一。

**vLLM**
开源 LLM serving 框架,Paged KV Cache 发明者。

**SGLang**
另一个开源 LLM serving 框架。

**Nsight Systems(nsys)**
NVIDIA 时间线 profiler,看 kernel 间关系。

**Nsight Compute(ncu)**
NVIDIA kernel 内部 profiler,看寄存器、SMEM、L2 等指标。

**compute-sanitizer**
CUDA 内存/同步/竞争检测工具,HPC-Ops 的 `make sanitizer` 集成。

## J. 路径速查

| 你想找 | 看这里 |
|---|---|
| 所有 Python API | `hpc/*.py` |
| C++ 注册到 PyTorch 的地方 | `src/*/entry.cc` |
| 真正的 CUDA kernel | `src/*/*.cu` 和 `*.cuh` |
| 单元测试 | `tests/test_*.py` |
| 性能基准 | `benchmark/*/README.md` |
| 构建配置 | `Makefile`, `setup.py`, `CMakeLists.txt` |
| 工具函数(SM count、TMA helper) | `src/utils/utils.cuh`, `src/utils/tma.cuh` |
| CUTLASS / CuTe 源码 | `3rd/cutlass/include/` |

---

返回 [README.md](./README.md) 查看全部章节。
