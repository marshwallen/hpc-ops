# 05 - Fused AllReduce + Residual + RMSNorm

> Tensor Parallel(TP)的 LLM 推理在每个 transformer block 末尾都要做 `AllReduce → +residual → RMSNorm` 这三步。HPC-Ops 用两种策略融合:**high-throughput**(基于 NVLink Multicast)适合 prefill 大 token,**low-latency**(基于 Lamport P2P + PDL)适合 decode 小 token。

## 1. 为什么要融合

标准 TP 推理在每个 block 末尾:

```
GPU 0: y0_local = MatMul_partial(x, W_partial)
GPU 1: y1_local = MatMul_partial(x, W_partial)
                          ↓
                AllReduce: y_full = sum(y_local)   ← 集合通信
                          ↓
                y_full = y_full + residual         ← 加法
                          ↓
                out = RMSNorm(y_full, gamma)       ← 归一化
```

朴素实现是 3 个独立 kernel(或者 AllReduce 是 NCCL 调用),问题:

1. **3 次 kernel launch**(NCCL 也算一次特殊 launch)
2. **2 次中间 HBM 往返**(y_full 写出再读回,y_full+residual 同理)
3. 在 small token 场景,这两项加起来可能比真正的"减法部分"还大

**融合**:把这 3 步压成 1 个 kernel,所有中间结果留在 register / shared memory,无中间 HBM IO。

## 2. 两种实现策略

| | High-Throughput | Low-Latency |
|---|---|---|
| 算法 | NVLink Multicast 单次广播 | Two-shot Lamport P2P |
| 适用 token 数 | 大(prefill 类) | 小(decode 类) |
| 通信特点 | 一次写到 multicast 地址,所有 rank 同时收 | rank 之间点对点写,通过 Lamport 算法保证一致性 |
| Kernel 数 | 1 | 2(scatter + broadcast) |
| 是否用 PDL | 否 | 是(两个 kernel 之间) |
| 入口 | `fuse_allreduce_rmsnorm_high_throughput` | `fuse_allreduce_rmsnorm_low_latency` |

## 3. High-Throughput: NVLink Multicast 路径

### 3.1 NVLink Multicast 速览

H100 NVLink Generation 4 + NVSwitch 提供 **multicast** 能力:

```
GPU0 写入 multicast address → 硬件自动复制到 GPU1/2/3/...

效果:
  - 一次写 = 多次接收
  - 节省总线带宽(原本要 N-1 次单播)
  - 完美匹配 AllReduce 的"广播求和"语义
```

CUDA 提供 multicast object API(`cuMulticastCreate / cuMulticastAddDevice / cuMulticastBindMem`)。详细看 `src/communicator/multicast_*.cc`。

### 3.2 算法

```
8 个 rank,每个 rank 持有 x_local: [N, H]:

Phase 0: 各 rank 在 local 算好 x_local 后,先做 sum_local = scale(x_local)

Phase 1: 每个 rank 把它的 x_local 的一段(N/world_size 行)
         通过 multicast 写,所有 rank 同时收到完整 y_full

Phase 2: 每个 rank 拿到 y_full 后,本地做 + residual + RMSNorm
         (这部分是普通 elementwise + reduction)

结果:
  - 通信只走 1 次 multicast(N*H 字节)
  - 计算和通信在 kernel 内部重叠
```

### 3.3 Python API

```python
import hpc

# 1. 创建 MulticastCommunicator(每个 rank 一份)
comm = hpc.MulticastCommunicator(
    rank, world_size, device_id=rank, comm_name="hpc_ar_ht_xxx"
)

# 2. 分配 multicast buffer(local + multimem view)
in_x, in_hdl = hpc.empty_multimem(comm, [N_pad, H], dtype=torch.bfloat16, device=device)
out_x, out_hdl = hpc.empty_multimem(comm, [N_pad, H], dtype=torch.bfloat16, device=device)

# 3. 把数据写进 local buffer
in_x[:N, :] = my_input

# 4. 调融合算子
hpc.fuse_allreduce_rmsnorm_high_throughput(
    in_x[start:end, :],                                       # 本 rank 负责的一段
    in_hdl.get_multimem_buff(in_x[start:end, :].shape, ...),  # 对应的 multimem view
    residual[start:end, :],
    weight,
    rms_norm_eps,
    signal=in_hdl.get_signal(rank, [max_blocks * world_size]).view(torch.int64),
    rank=rank,
    world_size=world_size,
    num_max_blocks=num_max_blocks,
    output_x=out_x[start:end, :],
    output_multicast_x=out_hdl.get_multimem_buff(out_x[start:end, :].shape, ...),
    output_residual=out_residual[start:end, :],
)
```

`signal` 张量用于 rank 间同步(每个 CTA 一个信号位)。

### 3.4 `MulticastHandle` 数据结构

`hpc/multicast_handle.py` 是一层 Python 封装:

```python
class MulticastHandle:
    def __init__(self, multicomm, size, dtype):
        # 分配 N+1 个对应的 buffer:
        #   org_buffer_dict_[0..world_size-1]: 各 rank 的 local view(可以远程读)
        #   org_buffer_dict_[-1]:               multimem ptr 的 view
        # 这些 buffer 共享底层物理内存(通过 multicast object 绑定)

    def get_buffer(self, rank, *sizes, dtype, storage_offset=0):
        # 取某个 rank 的 local 视图

    def get_multimem_buff(self, *sizes, dtype, storage_offset=0):
        # 取 multimem 视图(写它 = 广播)

    def get_signal(self, rank, *sizes, storage_offset=0):
        # 取 rank 的 signal 区(用于 CTA 间同步)
```

底层物理内存布局:

```
单个 rank 的对称分配区:
  [0 .. buffer_size]            data buffer
  [signal_offset .. end]        signal buffer (signal_offset 是 16-aligned 的 buffer_size)
```

### 3.5 Kernel 实现要点

源码 `src/allreduce/fuse_allreduce_rmsnorm_high_throughput.cu`(此处仅讲思想):

```cpp
__global__ void high_throughput_kernel(
    const T* x_local,         // 本 rank 的 x[start:end]
    const T* mc_input,        // 对应的 multimem 写入地址
    const T* residual,
    const T* weight,
    T* output_x,
    T* mc_output,
    T* output_residual,
    uint64_t* signal,
    int rank, int world_size, int num_tokens, int hidden_size, float eps)
{
    // 1. 每个 CTA 负责 hidden 的一段,每个 token 一组 thread
    // 2. 把 x_local 的本段 multicast 写出去(NVLink 硬件复制)
    multimem_write(mc_input + ..., x_local + ...);

    // 3. 等所有 rank 都写完(rank 间 signal 同步)
    if (last_thread) atomicAdd(signal + rank * num_blocks + blockIdx.x, 1);
    while (atomicLoad(signal + ...) < world_size) ;   // spin

    // 4. 从 local 视图读回完整的 y_full (= sum across ranks)
    //    (硬件 already done reduction during multicast)
    auto y = read_from_local_buffer();

    // 5. + residual
    y += residual;

    // 6. RMSNorm
    auto sum_sq = block_reduce_sum(y * y);
    auto inv_rms = rsqrt(sum_sq / hidden_size + eps);
    output = y * inv_rms * weight;

    // 7. Store
    output_x[...] = output;
    mc_output[...] = output;  // 同时 multicast 出去给下一层用
    output_residual[...] = y;
}
```

注:实际硬件是否在 multicast 阶段做 reduction 取决于 multicast 操作类型;HPC-Ops 使用的 multimem_*ld_reduce 类指令可以在 load 阶段完成 sum。

## 4. Low-Latency: Lamport P2P + PDL 路径

### 4.1 为什么 multicast 不适合小 token

Multicast 操作有固定启动开销(~5-10 μs)。对于 `num_tokens=8` 的 decode 场景,这个开销可能占总延迟一半以上。**P2P 直接拷贝**可能更快。

### 4.2 Two-Shot AllReduce

经典 ring AllReduce 需要 `2(N-1)` 步,多步累加。两轮(two-shot)算法在 N≤8 的小集群里更简单更快:

```
Stage 1 (Scatter):
  每个 rank 负责 hidden 的 1/N 段
  每个 rank 把它的 x_local 的对应段写到所有其他 rank 的同一段
  完成后:每个 rank 在自己负责的段上有完整的 N 份数据,可本地 sum

Stage 2 (Broadcast):
  每个 rank 把自己 sum 完的段广播给所有 rank
  完成后:所有 rank 都有完整 y_full
```

通信量:`2 * x_size * (N-1) / N`,接近最优。但是 P2P 需要解决一致性问题。

### 4.3 Lamport 算法

P2P 写入对方的 buffer 时,**接收方怎么知道数据写完了**?传统方法是 barrier/atomic flag,但开销大。

**Lamport 协议**(灵感来自 Lamport 时钟):

```
约定:正常数据不会出现"负零"(0x8000 for fp16/bf16)
约定:每次开始时,接收 buffer 初始化为 negative-zero
约定:发送方写正常数据,接收方 spin 等到读到的不是 negative-zero 时表示数据到达
```

无需显式 flag,**用数据本身做"魔法标记"**。代价:接收方需要 spin polling,但只 spin 几个 cycle 通常就到了。

源码:`src/allreduce/fuse_allreduce_rmsnorm_low_latency.h` 里有完整的 `LamportFlags` 和 `negZero<T>()` 工具:

```cpp
template <typename T>
static constexpr __device__ __host__ T negZero() {
    // float -> -0.0f
    // bf16  -> 0x8000U
    // half  -> 0x8000U
}
```

### 4.4 Lamport Buffer 3-stage 轮换

为了让"清理-写入-读取"并发,buffer 维护 3 个 stage 轮换:

```
       stage 0     stage 1     stage 2
       ────────    ────────    ────────
T=0    current     dirty       free
T=1    free        current     dirty
T=2    dirty       free        current
```

- **current**:本次 iter 正在写/读的 buffer
- **dirty**:上次 iter 用过的,需要清零(其他线程异步做)
- **free**:下次 iter 即将使用

`LamportFlags` 结构体的 `clearDirtyLamportBuf` 方法在前向 launch 时被调用,顺便把 dirty buffer 清回 negative-zero,**无需额外 kernel**。

### 4.5 PDL chain

两个 kernel(scatter 和 broadcast)之间用 **Programmatic Dependent Launch**:

```
Kernel 1 (Scatter):    launch (with PDL)
                       一旦 CTA 群开始往全局内存写,
                       Kernel 2 (Broadcast) 就被允许启动

Kernel 2 (Broadcast):  launch (depends on 1)
                       CTA 通过 mbarrier 等 stage1 数据到达
                       到达后立即开始下一步,无 kernel launch 间隔
```

这消除了"等 kernel 1 完全退出 SM 才能 launch kernel 2"的传统延迟。

### 4.6 Python API

```python
hpc.fuse_allreduce_rmsnorm_low_latency(
    input_x,             # 本 rank 的输入
    multicast_x,         # multimem 视图(虽然算法是 P2P,但需要 multimem 提供 buffer)
    data_buffer_ptrs,    # [world_size] int64,所有 rank 的 buffer 地址
    multinode_x,         # 本 rank 的 lamport buffer
    buffer_flags,        # lamport flag state
    world_size, rank,
    residual_in,
    weight_gamma,
    rms_norm_eps,
    num_max_blocks,
    output_x=None,       # 默认 in-place
    residual_out=None,
    launch_with_pdl=True,
)
```

### 4.7 与 high-throughput 的代码结构差异

`fuse_allreduce_rmsnorm_low_latency` 内部其实是**两个独立 kernel**(scatter + broadcast),通过 PDL 串起来。从用户角度看是一个调用。

`hpc/allreduce.py` 的 wrapper 显式传 `use_two_shot=True` 和 `launch_with_pdl=True`,这两个开关在底层暴露但默认值就是这样。

## 5. 性能数字(README + benchmark)

| 场景 | 对比 | 加速 |
|---|---|---|
| 单节点 BF16 TP,hidden 4096/5120/7168 | NCCL AllReduce + 朴素 RMSNorm | Up to 1.76x |
| 同上 | FlashInfer | 视 token 数取最优 |

复现:

```bash
cd benchmark/fuse_allreduce_rmsorm/
python3 bench_allreduce_rmsnorm.py \
  --hidden 7168 \
  --tokens 8 32 128 512 4096 8192 16384 32768 \
  --csv allreduce_rmsnorm.csv
```

输出字段:
- `hpc_ops_ht_us`:high-throughput 路径
- `hpc_ops_ll_us`:low-latency 路径
- `nccl_us`:NCCL baseline
- `flashinfer_us`:FlashInfer baseline
- `hpc_best_us` = `min(ht, ll)`(用户应根据 token 数自动切换)

## 6. 数据约束与硬件要求

- **硬件**:单节点 SM90 / H20 GPUs + NVLink 4 + NVSwitch
- **hidden_size**:必须是 `4096`, `5120`, `7168` 三个之一(kernel `TORCH_CHECK` 强约束)
- **dtype**:只支持 BF16
- **token_dim 对齐**:必须是 `sizeof(float4)/sizeof(bf16) = 8` 的倍数(向量化 load)
- **world_size**:2-64(实际单机 8 卡为主)

为什么 hidden 是固定值?因为 kernel 在编译期特化了 cluster size 和 SMEM 分配。要加新 hidden,改 dispatch + 编译。

## 7. 如何在现有框架里替换

vLLM 或 SGLang 的 attention block 末尾通常是:

```python
# old
y = nccl_allreduce(y_local)
y = y + residual
out = rmsnorm(y, gamma, eps)
```

替换为:

```python
hpc.fuse_allreduce_rmsnorm_high_throughput(
    in_x=y_local,
    multicast_x=mc_view_of_y_local,
    residual=residual,
    weight=gamma,
    rms_norm_eps=eps,
    signal=signal_buffer,
    rank=rank, world_size=world_size,
    num_max_blocks=132,  # H100 SM 数
    output_x=out_buffer,
    output_multicast_x=mc_view_of_out_buffer,
    output_residual=new_residual,
)
out = out_buffer
```

前置工作是用 `hpc.empty_multimem` 分配 mc buffer,通常在模型初始化时一次性完成。

## 8. 阅读源码路径

1. **从 entry 开始**:`src/allreduce/entry.cc`,看两个 wrapper 的张量验证。
2. **High-throughput kernel**:`src/allreduce/fuse_allreduce_rmsnorm_high_throughput.cu`,multicast + 同步信号。
3. **Low-latency 工具**:`src/allreduce/fuse_allreduce_rmsnorm_low_latency.h` 是核心,所有 Lamport / PackedVec / negZero / LamportFlags 都在头文件。
4. **Low-latency kernel**:`src/allreduce/fuse_allreduce_rmsnorm_low_latency.cu`,两个 kernel + PDL 串接。
5. **Communicator**:`src/communicator/`,看 multicast object 如何 bootstrap(TCP/Unix socket 引导 + CUDA multicast 绑定)。详见 [09_communicator.md](./09_communicator.md)。

---

下一篇:[06_sampler.md](./06_sampler.md) — Fused Sampler,把 decode 后的 top-k/top-p/temperature/penalty 全部压成 1-2 个 kernel。
