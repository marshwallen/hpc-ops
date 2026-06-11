# 09 - 单节点 NVLink Multicast 通信器

> HPC-Ops 的 fused allreduce 算子需要 multicast buffer,但**不依赖 NVSHMEM**(NVSHMEM 部署复杂、有许可证要求)。HPC-Ops 自研一套基于 **CUDA Multicast Object API + TCP/Unix Socket 引导通道**的通信器,纯 C++ 实现,通过 torch custom class 暴露给 Python。

> 本文件的内容偏"系统编程",对入门者来说理解一遍就好,不必深抠。但如果你想在自己的库里复刻同样的 multimem 支持,这一章是核心参考。

## 1. 为什么需要"通信器"

`fuse_allreduce_rmsnorm_*` kernel 工作时需要:
1. 一块**所有 rank 共享物理内存**的 multicast buffer
2. 知道每个 rank 在这块 buffer 里的虚拟地址(读/写对方的 buffer 时用)
3. 一个 multimem 地址(写它 = 广播到所有 rank)

要拿到这些,需要先在所有 rank 之间**协商**(交换 GPU 内存 export handle、device id 等)。这个协商需要一个 "out-of-band" 通道,通常用:

| 选择 | 优点 | 缺点 |
|---|---|---|
| MPI | 标准 | 重,依赖多 |
| NVSHMEM | 性能好 | 闭源、licensing |
| NCCL | 简单 | 不支持 multimem |
| **自写 TCP/Unix Socket** | 轻、可控、纯 C++ | 需自己实现协议 |

HPC-Ops 选第四种。

## 2. 类结构

```
hpc/communicator.py
  └─ MulticastCommunicator = torch.classes.hpc.MulticastCommunicator
     (这是一个 torch 注册的 C++ class,在 Python 里像普通对象用)

hpc/multicast_handle.py
  └─ MulticastHandle (纯 Python)
       依赖 MulticastCommunicator,封装了 multicast buffer 的分配与切片

src/communicator/  ← C++ 实现
  ├─ communicator.h/cc          - 底层 socket-based Communicator
  ├─ channel.h/cc               - TCP/Unix 通信通道
  ├─ listener.h/cc              - 接受 socket 连接
  ├─ connector.h/cc             - 主动建立 socket 连接
  ├─ protocol.h/cc              - 消息格式
  ├─ multicast_object_manager.h/cc  - CUDA multicast object 封装
  ├─ multicast_communicator.h/cc    - 把两者组合成"多播通信器"
  └─ entry.cc                       - torch class 注册
```

## 3. 启动流程

```python
# rank 0..world_size-1 各自:
comm = hpc.MulticastCommunicator(
    rank=rank,
    world_size=world_size,
    device_id=rank,            # 通常 GPU index == rank
    comm_name="my_app.sock",   # 决定 socket 路径/端口
)
```

发生了什么:

```
Step 1: 建立 control plane
  - rank 0 作为 root,起一个 Listener(Unix socket: /tmp/<comm_name>.sock,
    或 TCP: 由 comm_name 编码)
  - rank 1..N-1 作为 client,Connector 连到 root

Step 2: 全员同步基本信息
  - 各 rank 通过 Allgather 把 (rank, device_id, host_pid) 共享给所有人

Step 3: CUDA multicast object 创建
  - rank 0 调用 cuMulticastCreate 创建一个 multicast object
  - 通过 Allgather/Broadcast 把 multicast object 的 OS handle (fd) 发给其他 rank
    (Unix socket 通过 SCM_RIGHTS 传 fd)
  - 各 rank 通过 cuMulticastAddDevice 把自己的 device 加入到 multicast object

Step 4: 此时通信器就绪,可以开始 CreateTensorSync 分配 multimem buffer
```

注意 step 3 的 **fd 传递**是平台级技巧:`Allgather` 在 Linux 上需要走 Unix socket 才能传 fd(TCP 不行)。HPC-Ops 在底层做了双通道(Unix for fd, TCP for data)的兼容。

## 4. CreateTensorSync:分配 multimem buffer

```python
# 通过 Python wrapper(MulticastHandle 内部)
tensors = comm.CreateTensorSync(total_bytes)

# tensors 是 Dict[int, Tensor]:
#   tensors[0..world_size-1]: 各 rank 的 local view(同一物理内存)
#   tensors[-1]: multimem view (写它 = 广播)
```

底层 C++:

```cpp
// MulticastCommunicator::CreateTensorSync
bool CreateTensorSync(int64_t bytes,
                     std::vector<std::shared_ptr<void>>* sptrs,
                     std::vector<int>* devices,
                     std::shared_ptr<void>* multi_ptr,
                     int* multi_device)
{
    // 1. 本 rank 分配物理 GPU 内存(cuMemCreate)
    //    属性必须包含 multicast 支持
    //    得到 mem handle

    // 2. 通过 Communicator::AllgatherFd 交换所有 rank 的 mem handle (fd)
    //    现在每个 rank 都有 world_size 个 fd

    // 3. 对每个 fd:
    //    a. cuMemImportFromShareableHandle 导入到本进程
    //    b. cuMemMap 映射到本进程虚拟地址空间
    //    c. cuMemSetAccess 设置 RW 权限
    //    -> 得到 sptrs[r],每个 rank 的本进程映射地址

    // 4. cuMulticastBindMem 把所有 rank 的内存绑到 multicast object
    //    cuMemMap multicast handle 到本进程虚拟地址
    //    -> 得到 multi_ptr

    return true;
}
```

绑完之后:
- 写 `sptrs[r]` = 写 rank r 的 GPU(对方 rank)
- 写 `multi_ptr` = 写 multicast 地址(硬件广播)
- 读 `sptrs[self.rank]` = 读本地 GPU 内存

torch class 把这些原始指针包装成 `torch.Tensor`,但**用户不应该对它们做 torch 运算**,只能当作"内存指针容器"传给 HPC-Ops 的 kernel。

## 5. `MulticastHandle` 的进一步封装

`hpc/multicast_handle.py`:

```python
class MulticastHandle:
    def __init__(self, multicomm, size, dtype):
        # 1. 估算总大小:data + signal
        numel = product(size)
        buffer_size = numel * dtype.itemsize
        signal_offset = round_up(buffer_size, 16)
        signal_size = SIGNAL_SIZE     # 由 sm count 决定
        total_size = signal_offset + signal_size

        # 2. 一次性分配 total_size 字节的 multimem buffer
        self.org_buffer_dict_ = multicomm.CreateTensorSync(total_size)

        # 3. 拆分成 data part 和 signal part
        self.data_buffer_list_ = [
            self.org_buffer_dict_[i][:buffer_size] for i in range(world_size)
        ]
        self.signal_buffer_list_ = [
            self.org_buffer_dict_[i][signal_offset:] for i in range(world_size)
        ]
        self.multimem_data_buffer_ = self.org_buffer_dict_[-1][:buffer_size]
        self.multimem_signal_buffer_ = self.org_buffer_dict_[-1][signal_offset:]

        # 4. 把所有 rank 的 data 指针存到一个 int64 张量
        #    (kernel 直接用这个张量来 P2P 访问)
        self.data_buffer_ptrs_ = torch.tensor([
            self.data_buffer_list_[i].data_ptr() for i in range(world_size)
        ], dtype=torch.int64)
        self.data_buffer_ptrs_dev_ = self.data_buffer_ptrs_.cuda()
        # signal 同理
```

提供的方法:

- `get_buffer(rank, *sizes, dtype, storage_offset)`:取某个 rank 的 local data 视图
- `get_signal(rank, ...)`:取某个 rank 的 signal 视图(uint32)
- `get_multimem_buff(...)`:取广播视图
- `get_multimem_signal(...)`:取广播 signal 视图

## 6. Signal buffer 的作用

Allreduce kernel 需要在 rank 之间同步"我已经把数据写完了"。HPC-Ops 在每个 rank 的 multimem buffer **后部**预留一段 signal 区:

```
内存布局:
  [0 .. buffer_size]                 ← data
  [buffer_size .. signal_offset]     ← 16-byte align padding
  [signal_offset .. signal_offset + signal_size]   ← signal

signal_size = MAX_CUDA_P2P_DOMAIN_SIZE * num_sm * 4 bytes
            = 72 * 132 * 4 = ~38 KB (H100)
```

足够每个 CTA 一个 32 位 atomic 信号。kernel 用法:

```cpp
__device__ void wait_all_ranks(uint32_t* signal_local, int world_size, int cta_id) {
    if (last_thread) atomicAdd(&signal_local[cta_id], 1);
    while (atomicLoad(signal_local[cta_id]) < world_size) ;
}
```

这是 spin lock,但因为只等 ~几 μs(NVLink 写完成),CPU 占用可忽略。

## 7. 与 NCCL/NVSHMEM 的对比

| 项 | NCCL | NVSHMEM | HPC-Ops Multicast |
|---|---|---|---|
| 编程模型 | API(ncclAllReduce) | put/get 远程访问 | tensor + multimem 指针 |
| Multimem 支持 | 隐藏在内部 | 支持 | 显式暴露给 kernel |
| 跨节点 | 是 | 是 | 单节点(无 InfiniBand 支持) |
| 部署 | 需要 NCCL 库 | NVSHMEM + 注册 | 纯 C++ + CUDA driver |
| 灵活度 | 黑盒 | 灵活 | 灵活 |
| 性能 | 优 | 优 | 与 NVSHMEM 同档 |

HPC-Ops 的定位:**单节点 NVLink 域内**,通过自研通信器解锁 multimem 能力,不依赖外部库。如果需要跨节点,需要把 multicast 路径退化或换成 NCCL。

## 8. 测试方法:多进程 fork

由于 multicast 需要多个 rank 同时初始化,测试代码用 `multiprocessing.spawn`:

```python
# tests/test_fuse_allreduce_rmsnorm_high_throughput.py
def run_task(rank, world_size, N, H, num_max_blocks):
    torch.cuda.set_device(rank)
    
    # 每个子进程独立创建 communicator,通过 socket 自动握手
    comm = hpc.MulticastCommunicator(
        rank, world_size, rank,
        f"hpc_ar_ht_ws{world_size}_N{N}_H{H}",
    )
    
    in_x, in_hdl = hpc.empty_multimem(comm, [N, H], dtype=torch.bfloat16)
    # 跑算子...

def test_main():
    world_size = 8
    multiprocessing.spawn(
        run_task,
        args=(world_size, N, H, num_max_blocks),
        nprocs=world_size,
    )
```

`comm_name` 里编进了所有维度参数,避免不同测试用例的 socket 冲突。

## 9. 阅读源码路径

1. **`hpc/multicast_handle.py`**:Python 封装,理解 buffer 切片逻辑。
2. **`src/communicator/entry.cc`**:torch class 注册,看 Python ↔ C++ 边界。
3. **`src/communicator/multicast_communicator.h/cc`**:主要 API 实现。
4. **`src/communicator/multicast_object_manager.h/cc`**:CUDA multicast object 的具体 API 封装(`cuMulticastCreate`, `cuMulticastAddDevice`, `cuMulticastBindMem` 等)。
5. **`src/communicator/communicator.h/cc`** + `channel.h/cc` + `protocol.h/cc`:控制面 socket 通信(Allgather、Broadcast、AllgatherFd)。
6. **`src/communicator/listener.h/cc`** + `connector.h/cc`:socket 服务端 / 客户端。

入门顺序建议:先 entry.cc → multicast_communicator.cc → communicator.cc → 各子模块。

## 10. 上手示例

最小的 multimem hello world:

```python
import torch
import hpc
import multiprocessing as mp

def worker(rank, world_size):
    torch.cuda.set_device(rank)
    
    comm = hpc.MulticastCommunicator(rank, world_size, rank, "hello_mc")
    print(f"[rank {rank}] init OK, comm_rank={comm.GetRank()}")
    
    # 分配 1 MB 共享 buffer
    x, hdl = hpc.empty_multimem(comm, [1024, 1024], dtype=torch.bfloat16)
    
    # 每个 rank 在自己的 local 部分写 rank id
    x.fill_(float(rank))
    
    comm.Barrier()
    
    # 读另一个 rank 的 local view
    other_view = hdl.get_buffer(
        (rank + 1) % world_size, [1024, 1024], dtype=torch.bfloat16
    )
    print(f"[rank {rank}] read rank {(rank+1) % world_size} = {other_view[0,0].item()}")

if __name__ == "__main__":
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size)
```

期望输出:

```
[rank 0] init OK, comm_rank=0
[rank 1] init OK, comm_rank=1
[rank 0] read rank 1 = 1.0
[rank 1] read rank 0 = 0.0
```

如果你看到这个输出,说明 multicast 通信器 OK,后续可以放心调 fuse_allreduce 算子。

---

下一篇:[10_build_and_test.md](./10_build_and_test.md) — 详细的编译、测试、benchmark 流程。
