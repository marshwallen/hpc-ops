# HPC-Ops 技术文档

> 面向 **AI Infra 入门者** 的 HPC-Ops 深度技术解读。
>
> HPC-Ops 是腾讯混元 AI Infra 团队开源的、面向大模型推理热路径(Attention、MoE、GEMM、采样、归一化、通信-计算融合)的高性能算子库。

## 谁应该读这份文档

- 刚入门 AI Infra,听过 vLLM、SGLang、FlashAttention 等名字,但还没系统理解 LLM 推理为什么需要"手写算子"。
- 有 PyTorch 基础和 C/C++ 编程经验,但 CUDA、CUTLASS、CuTe 一知半解。
- 希望通过一个**真实的产线级算子库**来学习现代 GPU 编程范式(SM90 Hopper、Tensor Core、TMA、WGMMA、cp.async、PDL、Multicast)。

如果你已经是熟练的 GPU 内核工程师,这份文档对你来说会偏入门;此时建议直接读源码并参考 `benchmark/` 来理解每个算子的优化点。

## 文档目录

按照学习顺序排列,**强烈建议按顺序阅读**前面三篇。

### 0. 入门铺垫

| 文档 | 内容 | 适用读者 |
|---|---|---|
| [00_background.md](./00_background.md) | **LLM 推理与 GPU 编程背景**:Prefill/Decode、KV Cache、Tensor Parallelism、Expert Parallelism、Tensor Core、TMA、Warp Specialization、PDL 等基础概念图解。 | 完全的入门者 |

### 1. 项目结构

| 文档 | 内容 |
|---|---|
| [01_repository_structure.md](./01_repository_structure.md) | **仓库目录、构建系统、Python 与 C++/CUDA 的胶水层**。讲清楚一个 `import hpc` 是如何走到 CUDA kernel 的。 |

### 2. 算子分册

每个算子家族单独成篇。读完任意一篇你应该能:
- 用一句话讲清这个算子在 LLM 推理流程中所处的位置。
- 描述它的 Python API 输入/输出张量布局。
- 说出 2-3 个主要的优化技巧。

| 文档 | 算子家族 | 关键技术 |
|---|---|---|
| [02_attention.md](./02_attention.md) | Attention(Prefill / Decode / Paged KV Cache / FP8 / Block-Sparse / Dynamic Scheduling) | Warp Specialization, TMA, Split-K, Dynamic Bin-Packing |
| [03_gemm_and_group_gemm.md](./03_gemm_and_group_gemm.md) | GEMM(BF16×FP32 双精度合成)、Grouped GEMM(FP8 per-tensor / block-wise) | CuTe + WGMMA, 双 BF16 模拟 FP32, cp.async |
| [04_fused_moe.md](./04_fused_moe.md) | Fused MoE(per-tensor / block-wise FP8) | Gate-Up 直接路由、cp.async pipeline、PDL |
| [05_fused_allreduce_rmsnorm.md](./05_fused_allreduce_rmsnorm.md) | AllReduce + Residual + RMSNorm 融合(high-throughput / low-latency) | NVLink Multicast、Lamport P2P、两阶段 AllReduce |
| [06_sampler.md](./06_sampler.md) | Fused Sampler(repetition penalty / temperature / top-k / top-p / Gumbel-max) | Cluster-Cooperative top-k、温度快速路径、curand |
| [07_normalization_rope_activation.md](./07_normalization_rope_activation.md) | RMSNorm 量化、RoPE+KV Store、Activation+SiLU+量化 | 单 kernel 内融合多个轻量层 |
| [08_stem_sparse_attention.md](./08_stem_sparse_attention.md) | Stem 块稀疏注意力(OAM 评分 + TPD 选块) | 长上下文稀疏化、块稀疏掩码生成 |
| [09_communicator.md](./09_communicator.md) | 单节点 NVLink Multicast 通信器(无 NVSHMEM 依赖) | CUDA Multicast Object、TCP/Unix 引导通道 |

### 3. 实践

| 文档 | 内容 |
|---|---|
| [10_build_and_test.md](./10_build_and_test.md) | **从源码构建、运行测试、复现 benchmark、写新算子**。包含 `compute-sanitizer` 用法,以及如何在 vLLM / SGLang 中集成 HPC-Ops。 |
| [11_glossary.md](./11_glossary.md) | **术语表**。一次性查清缩写(BF16, FP8, GQA, MoE, NHD/HND, TMA, WGMMA, PDL, NVLink, Multicast, Lamport, …)。 |

## 阅读建议

1. **如果你是完全的新手**:先把 [00_background.md](./00_background.md) 通读一遍,然后挑一个你感兴趣的算子(推荐 Attention)从 [02_attention.md](./02_attention.md) 开始,边看边对照 `tests/` 下的测试代码运行起来。
2. **如果你已经懂 PyTorch、CUDA 基础,只是想看 HPC-Ops 怎么用**:直接从 [01_repository_structure.md](./01_repository_structure.md) 开始,然后用 [10_build_and_test.md](./10_build_and_test.md) 跑起来。
3. **如果你只想了解某一个算子**:从 [README.md](./README.md) 的目录直接进入对应文档,每个算子文档都是自洽的。

## 与官方 README 的关系

仓库根目录的 [`README.md`](../README.md) 是**面向用户的项目主页**:介绍亮点、性能数字、Quick Start。

`docs/` 下的文档是**面向想读懂代码、扩展代码的人**:更长、更深、更多原理图解。两者互补。
