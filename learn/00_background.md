# 00 - LLM 推理与 GPU 编程背景

> 本文为完全新手准备。如果你已经清楚 Prefill/Decode、Tensor Core、WGMMA 这些名词,直接跳到 [01_repository_structure.md](./01_repository_structure.md)。

阅读这份文档不需要预先看过任何论文。读完后,你应该能用自己的话回答这些问题:

- 为什么 LLM 推理可以拆成两个阶段?
- KV Cache 占多大?为什么要"分页"?
- Tensor Core 比 CUDA Core 强在哪里?
- 为什么 NVIDIA 不停推新数据格式(BF16、FP8、FP4)?
- 一个"算子融合"的 kernel 究竟省了什么?

---

## 1. LLM 推理:一个简化模型

把一个大语言模型(decoder-only Transformer)想象成一台**逐 token 输出**的状态机:

```
用户输入: "今天天气"
              ↓
          [模型]
              ↓
输出 token: "真"     ← 这一步叫一次 forward
                            (输出 1 个 token)
              ↓
拼接: "今天天气真"
              ↓
          [模型]
              ↓
输出 token: "好"
              ↓
拼接: "今天天气真好"
              ↓
          [模型]
              ↓
输出 token: "</eos>"   ← 结束符,停止
```

显然有一个明显的优化机会:每次 forward,前缀里大部分 token 都和上一次一样,**为什么要重算一遍?**

### 1.1 Prefill 和 Decode

为了利用这个性质,推理框架把请求处理拆成两个阶段:

| 阶段 | 输入长度 | 输出长度 | 性质 |
|---|---|---|---|
| **Prefill**(预填充) | 整段 prompt(可以很长) | 1 个 token + 缓存 | **大矩阵乘**,计算密集 |
| **Decode**(自回归解码) | 1 个 token(MTP 时是 2-4 个) | 1 个 token + 更新缓存 | **小矩阵乘 + 长序列读**,**访存密集** |

**Prefill 例子**:用户输入 "请帮我写一首诗"(假设 8 个 token),模型一次性吃下这 8 个 token,跑一遍 forward,得到 8 个位置的中间状态(KV)以及第 9 个 token。

**Decode 例子**:接下来每一步只输入第 N 个 token,但模型仍然需要在 attention 里和前面所有 N-1 个位置交互。所以前面 N-1 个位置的 K 和 V 必须**缓存**起来。这就是 **KV Cache**。

### 1.2 KV Cache 有多大?

以 LLaMA-7B 为例,假设 `num_layers=32, num_kv_heads=32, head_dim=128, bf16`:

```
每 token KV Cache 大小 = 2 × num_layers × num_kv_heads × head_dim × 2 bytes
                       = 2 × 32 × 32 × 128 × 2
                       = 524,288 bytes ≈ 0.5 MB
```

一个 32K 上下文请求,KV Cache 就是 **16 GB**。一张 80GB H100 顶多能装 5 条请求。这就是为什么"显存"和"吞吐量"在 LLM serving 里是同义词。

### 1.3 KV Cache 为什么要分页

传统的实现给每条请求预分配一段连续显存,长度按 max_seq_len(比如 32K)分配。问题:实际平均请求长度可能只有 2K,**显存碎片 + 利用率低**。

**Paged KV Cache**(vLLM 首创,现在是事实标准):

```
显存里维护一个 KV Cache 池,切成固定大小的"块"(比如 64 个 token 一块):

    ┌───┬───┬───┬───┬───┬───┬───┬───┐
显存 │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ ... 每块 64 token
    └───┴───┴───┴───┴───┴───┴───┴───┘

每条请求维护一张"块 ID 表"(block_ids),按需要分配新块:

请求 A (300 token): block_ids = [3, 1, 7, 4, 0]   (5 个块 = 320 token, 多余的 padding)
请求 B (90 token):  block_ids = [2, 6]            (2 个块 = 128 token)
```

这就是 HPC-Ops 里所有 attention 算子都接 `kcache: [num_blocks, block_size, num_head_kv, num_dim]` + `block_ids: [num_batch, max_blocks]` 的根本原因。

### 1.4 GQA: Grouped Query Attention

完整 multi-head attention(MHA)里 Q 头和 K 头数量相同。但实测发现 KV 可以共享:**每 G 个 Q 头共享一组 KV**,KV Cache 缩小 G 倍,性能损失可忽略。这就是 **GQA**(Grouped Query Attention)。

HPC-Ops 里 `num_head_q / num_head_kv` 这个比值就是 GQA 组大小,常见 4 或 8。当 KV head=1 时退化为 MQA(Multi-Query Attention),特别是 decode 阶段的访存大户。

---

## 2. GPU 体系结构速览

HPC-Ops 主要面向 NVIDIA SM90(Hopper 架构,H100/H20/H200/GH200)。理解以下几个概念后你才能看懂源码注释。

### 2.1 SM、Warp、CTA、Cluster

```
GPU
 ├─ SM 0
 │   ├─ Warp 0   ─┐
 │   ├─ Warp 1    │  每个 SM 同时跑多个 Warp,
 │   ├─ Warp 2    │  每个 Warp 32 个线程,
 │   └─ Warp 3   ─┘  4 个 Warp = 1 Warp Group(WG)
 ├─ SM 1
 ├─ ...
 └─ SM 131 (H100 SXM 有 132 个 SM)

Kernel 启动时:
  grid = N 个 CTA(Cooperative Thread Array,也叫 thread block)
  每个 CTA 调度到一个 SM 上执行
  一个 SM 可以同时驻留 1-N 个 CTA(看资源)

SM90 新增 Cluster(线程块簇):
  一个 Cluster 内的 CTA 可以
    - 共享 distributed shared memory(DSMEM)
    - 互相 cluster barrier 同步
  Cluster 大小最多 8 个 CTA(在 1 个 GPC 内)
```

为什么要 Cluster?因为 SM90 的 TMA + WGMMA 操作粒度很大,单个 CTA 的 shared memory 不够装,需要在 cluster 里**协作存储**。HPC-Ops 的 sampler topk 就用了 cluster。

### 2.2 Tensor Core 与 WGMMA

**普通 CUDA Core**:一个线程一拍做一次乘加,SIMD 风格。

**Tensor Core**(从 Volta 开始):一个 warp 一拍做一次 16×16×16 矩阵乘加(MMA)。速度提升一个数量级。

**SM90 WGMMA**(Warpgroup MMA,新指令):4 个 warp(128 线程)合作一拍做更大尺寸的矩阵乘加,例如 `m64n128k32 fp8` 一条指令搞定。**支持异步**:发出指令后线程可以继续干别的活,等结果时再 `wgmma.wait`。

意义:WGMMA 让"算"和"搬"可以重叠,这是现代 attention kernel 的核心优化点。

```
传统流水线(三阶段不能重叠):
  load → compute → store

WGMMA + TMA + WS(后面讲) 流水线:
  ┌─ TMA load (异步) ──→
  ├─ WGMMA compute (异步) ──→
  └─ TMA store (异步) ──→
  全部并行,只在边界同步
```

### 2.3 TMA: Tensor Memory Accelerator

SM90 引入的硬件单元,**专门负责显存↔shared memory 的数据搬运**。

```
传统 cp.async(SM80):每个线程发起搬运请求,需要凑 4-16 个线程/16 字节才高效。

TMA:    一条指令搬一整块 tile(比如 128×64),自动处理边界、stride、swizzle。
        提交后线程可以立刻干别的,通过 mbarrier 等待完成。
```

**为什么 HPC-Ops 反复出现 `tmas` 张量?** 因为 TMA 指令需要一个**描述符**(tensor descriptor,128 字节),里面编码了基地址、shape、stride、swizzle 模式。HPC-Ops 算子在 host 端预构造好描述符放在 `tmas` 张量里,kernel 启动时只需要把描述符地址传进去。

### 2.4 Warp Specialization (WS)

把一个 CTA 内的 4 个 warpgroup 分成两类:

- **Producer Warpgroup**:专门发起 TMA load,把数据搬进 shared memory。
- **Consumer Warpgroup**:专门做 WGMMA 计算和写回。

它们通过 shared memory 上的 **mbarrier** 通信:Producer 装满一格就 arrive,Consumer 等 wait 后开始算。

```
Producer: ─load tile 0─arrive(0)──load tile 1─arrive(1)──load tile 2─arrive(2)──
Consumer: ─wait(0)─compute tile 0──wait(1)─compute tile 1──wait(2)─compute tile 2─

效果: load 和 compute 完全重叠,IO 不再 stall 算。
```

HPC-Ops 的 prefill attention(`src/attention/prefill/warp_spec_*.cu`)、group GEMM 等都用 WS 范式。

### 2.5 cp.async

SM80 引入的异步拷贝指令(`cp.async`)。WS 出现前是主流;HPC-Ops 在某些场景下(低延迟 MoE)发现**不用 WS,改用 cp.async + 让更多 CTA 驻留 SM**,延迟更低。这就是 `src/group_gemm/cp_async/` 这个子目录的存在原因。

> "把流水线隐藏放在跨 CTA 的硬件调度层面,而不是 CTA 内的软件层面" —— 这是 HPC-Ops 的一个核心 insight。

### 2.6 PDL: Programmatic Dependent Launch

SM90 新特性。两个相继发射的 kernel,如果第二个的输入恰好是第一个的输出,**第二个 kernel 可以提前启动**,在第一个的 CTA 还没全部退出时就开始读还未完全写完的数据(只要内存依赖满足)。

```
传统串行启动(kernel B 必须等 A 全部结束):
  A:  [CTA0][CTA1][CTA2][CTA3]
  B:                            [CTA0][CTA1]...

PDL 启动(B 一启动就开跑,只在数据依赖上等):
  A:  [CTA0][CTA1][CTA2][CTA3]
  B:               [CTA0][CTA1]...   ← 早启动了 A 的尾巴时间
```

效果:**消除 kernel 之间的"气泡"**。HPC-Ops 的 fused MoE 用 PDL 把多个子 kernel 串起来。

### 2.7 NVLink Multicast

NVLink 是 GPU 间高速互连(900 GB/s @ H100)。新硬件支持 **multicast**:一个 GPU 写一个 multicast 地址,所有订阅了这个地址的 GPU 同时收到。

```
传统 AllReduce(ring 算法):
  GPU0 → GPU1 → GPU2 → GPU3 → GPU0 ... (4 个 hop)

Multicast AllReduce:
  GPU0 ┐
  GPU1 ├─→ multicast addr ─→ 所有 GPU 同时收到
  GPU2 ┤
  GPU3 ┘
```

HPC-Ops 的 `fuse_allreduce_rmsnorm_high_throughput` 走 multicast 路径,大 token 数下吞吐打爆传统 NCCL。

---

## 3. 数据类型:为什么会有这么多浮点格式

| 格式 | bits | 范围 | 精度 | 何时用 |
|---|---|---|---|---|
| FP32 | 32 | ±3.4e38 | 7 位 | 训练 master 权重、关键累加 |
| TF32 | 19 (硬件 32-bit slot) | ±3.4e38 | 10 位尾数 | Ampere Tensor Core 训练 |
| BF16 | 16 | ±3.4e38 | 7 位尾数(同 FP32) | 现代训练 + 推理默认 |
| FP16 | 16 | ±65504 | 10 位尾数 | 老式训练,有溢出风险 |
| FP8 E4M3 | 8 | ±448 | 3 位尾数 | 推理(权重 + 激活) |
| FP8 E5M2 | 8 | ±57344 | 2 位尾数 | 推理(累加器、scale 多) |
| INT8 | 8 | -128..127 | 整数 | 量化推理 |
| FP4 | 4 | 极少 | 极少 | Blackwell 新特性 |

**入门要点**:
- BF16 是当下 LLM 推理"默认精度"。
- FP8 是 SM90/SM100 的主力**推理**精度,需要做 per-tensor / per-token / block-wise scale 来补偿精度损失。
- HPC-Ops 大量算子提供 BF16 / FP8 双版本,这是为什么 `attention.py` 里有 `attention_decode_bf16` 和 `attention_decode_fp8` 两套。

### 3.1 量化方案的"粒度"

```
per-tensor:    整个张量一个 scale (1 个 float)
per-token:     每行(每个 token)一个 scale
per-head:      每个 attention head 一个 scale
block-wise:    每个 128-元素的小块一个 scale
```

精度从低到高,显存开销和计算开销也从低到高。HPC-Ops 通过 `QuantType` 枚举挑选不同的组合(见 `hpc/attention.py`)。

---

## 4. 算子融合(Kernel Fusion):为什么重要

考虑一个 transformer block 末尾的常见序列:

```
y_local = MatMul(x, W)          # GEMM
y_full  = AllReduce(y_local)    # 集合通信,跨 GPU
y_full  = y_full + residual     # 加法
out     = RMSNorm(y_full, gamma)  # 归一化
```

每一步都是一个 kernel,意味着:
- 4 次 kernel launch overhead(~5 μs 每次)。
- 中间结果 `y_local, y_full, y_full + residual` **都要写回 HBM 再读回来**。

如果在 5K hidden_dim、batch 64 上,中间结果一次 IO 就是 `64×5K×2 bytes ≈ 640 KB`,HBM 带宽 3TB/s 也要 ~200 ns/次,加上 4 个 kernel 共 ~1 μs 浪费在 IO 上。

**融合后**:把 AllReduce + Add + RMSNorm 写进**一个 kernel**,中间结果留在 register / shared memory 里,**零 HBM 中间 IO**,kernel launch 也只剩 1 次。

HPC-Ops 的命名套路:`fuse_xxx_yyy_zzz` 就是融合算子的标记。

---

## 5. 总结:为什么需要 HPC-Ops 这种库

通用框架(PyTorch、TensorFlow)的算子是"通用 + 单一职责":一个 MatMul、一个 Softmax、一个 RMSNorm。

LLM 推理走的是**极限优化路径**:
- 已知输入 shape 范围、数据类型、内存布局,可以**编译期特化**。
- 已知算子之间的依赖关系,可以**融合**。
- 已知硬件(SM90)的所有特性,可以全部用上(TMA、WGMMA、WS、PDL、Multicast)。

写这种 kernel 的门槛极高:CUDA + CuTe + CUTLASS + 性能调优经验。HPC-Ops 把这些算子作为"产线验证过的零件"开源出来,你既可以**直接拿来用**,也可以**作为现代 CUDA 编程的教学样本**。

---

下一篇:[01_repository_structure.md](./01_repository_structure.md) —— 仓库结构 + 构建流程 + Python/C++ 胶水层是怎么搭起来的。
