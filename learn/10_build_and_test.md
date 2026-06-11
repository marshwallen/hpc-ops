# 10 - 构建、测试与 Benchmark

> 本文是**实操手册**:从零到能跑测试、跑 benchmark、调试 CUDA 内核问题。

## 1. 环境准备

### 1.1 硬件

- NVIDIA SM90 GPU(Hopper 架构,H100 / H20 / H200 / GH200)
- 至少 80 GB 显存(测试覆盖率,benchmark 大场景需要)
- NVLink + NVSwitch(跑 allreduce benchmark 需要 8 卡同节点)

### 1.2 软件

```bash
# 必装
- Linux x86_64 (推荐 Ubuntu 22.04 或 Rocky 9)
- CUDA Toolkit 12.8 或更高
- Python 3.8 或更高
- gcc/g++ 11+ (支持 C++17)
- cmake 3.18+
```

### 1.3 Python 包

```bash
pip install -r requirements-dev.txt
```

`requirements-dev.txt` 关键依赖:
- `torch==2.7.0`(注意:HPC-Ops 的 `.abi3.so` 链接到 libtorch,版本必须严格匹配)
- `numpy==2.3.1`
- `pytest==8.4.1`
- `build==1.4.0`(打 wheel 用)
- `cpplint==2.0.2` / `clang-format==20.1.8`(格式化)

### 1.4 验证环境

```bash
# 1. 看 CUDA
nvcc --version    # 应显示 12.8+
nvidia-smi        # 应显示 H100/H20/...

# 2. 看 PyTorch
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 应输出: 2.7.0 True

# 3. 看 GPU 架构(应是 9.0)
python3 -c "import torch; print(torch.cuda.get_device_capability())"
# 应输出: (9, 0)
```

## 2. 编译

### 2.1 基本构建

```bash
cd hpc-ops

# 增量编译(开发用)
make all
# 等价 python3 setup.py build
# 输出在 build/lib.linux-x86_64-cpython-3X/hpc/

# 打 wheel(发布或安装用)
make wheel
# 输出在 dist/hpc_ops-*.whl

# 安装(全局)
pip install dist/*.whl
```

首次编译耗时 5-10 分钟(取决于 CPU 核数,`-j16` 默认),后续增量编译快很多。

### 2.2 看具体编译错误

```bash
# 直接调 cmake 看完整错误
mkdir -p build
cd build
cmake ..
cmake --build . --config Release -j16 2>&1 | tee build.log
```

常见错误:
| 错误 | 原因 | 解决 |
|---|---|---|
| `unsupported GPU architecture 'compute_90a'` | CUDA 版本太低 | 升级到 12.8+ |
| `cute/tensor.hpp: No such file` | CUTLASS 源码缺失 | 检查 `3rd/cutlass/include/` 是否完整 |
| `c10::Device` 链接错 | PyTorch 版本不匹配 | 装 torch==2.7.0 |
| `_GLIBCXX_USE_CXX11_ABI` 链接错 | 用了非官方 PyTorch | 用官方 pip 的 torch |

### 2.3 编译选项调优

`CMakeLists.txt` 默认开 `-DNDEBUG`,如果想 debug:

```bash
# 改一行 CMakeLists.txt
set(CMAKE_BUILD_TYPE "Debug")    # 原本是 "Release"

# 或在 cmake 命令行覆盖
cmake -DCMAKE_BUILD_TYPE=Debug ..

# 这会:
#  - 关掉 NDEBUG,启用 TORCH_CHECK 详细信息
#  - 加上 -G(devicedebug,可以 cuda-gdb)
#  - 减少优化级别
```

## 3. 测试

### 3.1 一键全跑

```bash
make test
# 等价:
#   for test in tests/test_*.py; do
#     python3 -m pytest -v --no-header --disable-warnings $test || exit 1
#   done
```

### 3.2 单文件 / 单 case

```bash
# 跑某算子的所有 case
python3 -m pytest -v tests/test_attention_decode_bf16.py

# 用 -k 过滤(支持 substring)
python3 -m pytest -v tests/test_attention_decode_bf16.py -k "num_batch1 and head_dim128"

# 跑某个具体参数化 case(从 -v 输出复制 id)
python3 -m pytest -v "tests/test_attention_decode_bf16.py::test_attention_decode_bf16[True-True-128-(1,4)-64-1024-1-1]"
```

### 3.3 测试的查找路径

测试代码里所有都有:

```python
sys.path.insert(0, os.path.realpath(
    list(Path(__file__).parent.glob("../build/lib.*/"))[0]
))
```

这意味着测试**直接引用 build/ 目录下的本地构建**,**不需要 pip install**。开发时只需 `make all`,测试就能看到最新代码。

### 3.4 compute-sanitizer 包装

`conftest.py` 提供一个高级特性:**自动用 compute-sanitizer 包装每次算子调用**。开启方法:

```bash
# 启用 memcheck + synccheck + racecheck(慢 10-20 倍)
make sanitizer

# 或单独开
SANITIZER_CHECK=memcheck python3 -m pytest -v tests/test_xxx.py
SANITIZER_CHECK=racecheck,synccheck python3 -m pytest -v tests/test_xxx.py
```

它会:
1. 每次 hpc 调用前后,把输入/输出 dump 到 `/dev/shm/tmp_hpc_*.pth`
2. 生成一个独立的 reproducer python 脚本
3. 用 `compute-sanitizer --tool=<check>` 跑这个脚本
4. 任何 sanitizer 错误都会失败

适合在 PR 之前作为"严格检查"运行。

### 3.5 Sanitizer 的常见检测

| Tool | 检测内容 |
|---|---|
| `memcheck` | 越界访问、未初始化读、释放后使用 |
| `racecheck` | shared memory 数据竞争 |
| `synccheck` | warp/cluster 同步原语错配 |
| `initcheck` | 寄存器未初始化使用 |

## 4. Benchmark

`benchmark/` 下每个子目录是一个独立 benchmark,都有自己的 README。基本套路:

```bash
cd benchmark/<area>
python3 <benchmark_script.py> [options]
```

### 4.1 Attention Decode Benchmark

```bash
python3 benchmark/attention_decode/bench_attention_decode_fp8.py \
  --timing nsys \
  --output-dir attention_decode_nsys \
  --csv attention_decode_fp8.csv
```

关键参数:
- `--timing nsys`:用 Nsight Systems 抓 NVTX range 取精确延迟(release 模式)
- `--timing event`:用 CUDA event 快速测(开发模式)
- `--check`:开启 dense 路径作为正确性参考

### 4.2 Sampler Benchmark

```bash
python3 benchmark/sampler/benchmark_sampler.py \
  --timing nsys \
  --output-dir sampler_nsys
```

输出对比 `hpc-ops` / `torch (vLLM style)` / `flashinfer`。

### 4.3 Route GEMM Benchmark

```bash
python3 benchmark/route_gemm/benchmark_gemm_bf16xfp32.py \
  --n 192 --k 4096 \
  --m-list 2,4,8,16,48,96,208,512,1024,2048,4096 \
  --timing nsys
```

对比 cuBLAS FP32 / cuBLAS TF32。

### 4.4 Fused MoE Benchmark

需要可选的 vLLM/SGLang 安装才能完整对比:

```bash
export VLLM_ROOT=/path/to/vllm
export SGLANG_ROOT=/path/to/sglang

python3 benchmark/fused_moe/benchmark_fuse_moe.py \
  --tp 8 --ep 1 \
  --providers hpcops vllm vllm_cutlass sglang \
  --csv fused_moe_tp8_ep1.csv
```

如果 vLLM/SGLang 没装,benchmark 会自动跳过它们,只跑 hpcops。

### 4.5 Fused AllReduce Benchmark

需要 **8 卡 NVLink 节点**:

```bash
cd benchmark/fuse_allreduce_rmsorm/
python3 bench_allreduce_rmsnorm.py \
  --hidden 7168 \
  --tokens 8 32 128 512 4096 8192 16384 32768 \
  --csv allreduce_rmsnorm.csv
```

benchmark 自己 spawn 8 个 worker 进程,**不需要 torchrun**。

## 5. 调试技巧

### 5.1 nsys profiler

抓 kernel 时间线:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=my_run \
  --capture-range=cudaProfilerApi \
  python3 my_script.py
```

然后用 `nsys-ui my_run.nsys-rep` 在桌面打开看时间线。

### 5.2 ncu kernel profiler

抓单个 kernel 的硬件指标(IPC、occupancy、HBM 带宽):

```bash
ncu --set full \
    -k regex:"attention_decode" \
    -o my_kernel_profile \
    python3 my_script.py
```

然后 `ncu-ui my_kernel_profile.ncu-rep`。

### 5.3 dump 中间张量

PyTorch 端容易:

```python
import torch
torch.save({"q": q, "k": k, ...}, "debug.pth")

# 另一进程
data = torch.load("debug.pth")
out = hpc.attention_decode_bf16(data["q"], ...)
```

CUDA kernel 端要 dump 中间张量需要插桩:在 kernel 内调用 `printf` 或写一个 debug buffer。HPC-Ops 没有内置 dump 机制,需要自己加。

### 5.4 cuda-gdb

```bash
# 1. 编译开 debug
# 改 CMakeLists.txt 加 -G,见 §2.3

# 2. 跑
cuda-gdb python3
(cuda-gdb) run my_script.py
(cuda-gdb) cuda kernel break attention_decode_kernel
(cuda-gdb) continue
```

cuda-gdb 性能很低,只适合复现复杂 bug。

## 6. 编写新算子的 checklist

如果你想给 HPC-Ops 加一个新算子(比如 `my_op`):

### 6.1 Python 接口

```python
# 新建 hpc/my_op.py
import torch
from torch import Tensor

def my_op(x: Tensor, ...) -> Tensor:
    """完整 docstring(像现有算子一样写清楚 shape 和 dtype)"""
    return torch.ops.hpc.my_op(x, ...)

@torch.library.register_fake("hpc::my_op")
def my_op_fake(x, ...):
    return torch.empty_like(x)
```

无需修改 `__init__.py` —— `_discover_modules` 会自动 import。

### 6.2 C++ entry

```cpp
// src/my_op/entry.cc
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/library.h>
#include "src/my_op/my_op.h"

namespace hpc::my_op {

torch::Tensor my_op_entry(const torch::Tensor& x, ...) {
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    TORCH_CHECK(x.device().is_cuda(), ...);
    // ... 验证 ...
    auto y = torch::empty_like(x);
    my_op_async(y.data_ptr(), x.data_ptr(), ..., stream);
    return y;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hpc, m) {
    m.def("my_op(Tensor x, ...) -> Tensor");
    m.impl("my_op", torch::kCUDA, &hpc::my_op::my_op_entry);
}
```

### 6.3 CUDA implementation

```cpp
// src/my_op/my_op.h
namespace hpc::my_op {
void my_op_async(void* y, const void* x, ..., cudaStream_t stream);
}

// src/my_op/my_op.cu
#include "src/my_op/my_op.h"
namespace hpc::my_op {

__global__ void my_op_kernel(...) { ... }

void my_op_async(void* y, const void* x, ..., cudaStream_t stream) {
    dim3 grid(...), block(...);
    my_op_kernel<<<grid, block, 0, stream>>>(...);
}

}  // namespace
```

### 6.4 测试

```python
# tests/test_my_op.py
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.realpath(
    list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import torch
import hpc

def test_my_op_basic():
    x = torch.randn(10, 20, device="cuda")
    y_ref = my_torch_reference(x)
    y_hpc = hpc.my_op(x)
    assert torch.allclose(y_ref, y_hpc, atol=1e-3)
```

### 6.5 重新编译 + 测试

```bash
make all
python3 -m pytest -v tests/test_my_op.py
```

CMake 的 `file(GLOB_RECURSE ...)` 会自动捡到新的 `.cu/.cc`。

### 6.6 格式化

```bash
make format
# 跑 black + clang-format + cpplint
```

PR 之前必跑。

## 7. 集成到 vLLM / SGLang

`benchmark/fused_moe/backends/hpcops.py` 是非常好的集成参考。基本套路:

```python
# 替换 vLLM 的 fused_moe 实现
import hpc

def vllm_fused_moe_hpcops(x, gate_up_w, down_w, ..., topk_ids, topk_weights, ...):
    # 1. 准备 scale 张量(vLLM 用的是 per-tensor static scale,直接复用)
    # 2. 准备 EP rank 信息
    # 3. 调 hpc.fuse_moe_blockwise(...) 或 hpc.fuse_moe_pertensor_fp8(...)
    return hpc.fuse_moe_blockwise(x, x_scale, gate_up_w, gate_up_scale, ...)

# Monkey patch
import vllm.model_executor.layers.fused_moe as vllm_moe
vllm_moe.fused_moe = vllm_fused_moe_hpcops
```

也可以走 vLLM 的 plugin 机制(更正式)。

## 8. 故障排查 cheatsheet

| 现象 | 可能原因 | 排查 |
|---|---|---|
| `_C.abi3.so: undefined symbol` | torch 版本不匹配 | `pip install torch==2.7.0 --force-reinstall` |
| `illegal instruction` 运行时 | GPU 不是 SM90 / 用了 sm_90 编译 | 确认 `--gpu-architectures=90a` |
| `out of resources` launch 失败 | SMEM 超过 228KB | 减小 tile size 或 stage 数 |
| `unspecified launch failure` | 内部访存越界 | 跑 `make sanitizer` 定位 |
| `RuntimeError: CUDA error: invalid argument` | tensor 不在 cuda / 不连续 | 检查 `x.is_contiguous() and x.device.type == 'cuda'` |
| Multimem allreduce 卡住 | rank 数不一致 / socket 名冲突 | 检查 `comm_name` 唯一性 |
| Allreduce 结果错 | hidden_size 不在 {4096, 5120, 7168} | 改输入 hidden 或扩展 kernel dispatch |
| `permission denied` 创建 socket | `/tmp` 没权限 / 容器隔离 | 改用 TCP 形式的 comm_name |

## 9. 性能调优入门

如果你跑出来的 hpc-ops 性能不如预期:

### 9.1 确认环境

```bash
# 1. 关掉 ECC(数据中心 GPU 可调)
nvidia-smi -e 0

# 2. 锁定 GPU 频率(避免 boost 抖动)
nvidia-smi -i 0 -lgc 1980,1980    # H100 SXM 最大

# 3. 关闭 MPS(可能有干扰)
echo quit | nvidia-cuda-mps-control

# 4. 进程独占 GPU
CUDA_VISIBLE_DEVICES=0 python3 my_script.py
```

### 9.2 用 nsys 看哪里慢

```bash
nsys profile --trace=cuda,nvtx -o profile python3 my_script.py
nsys stats profile.nsys-rep
```

看每个 kernel 的占比、launch 间隔、是否有 idle gap。

### 9.3 用 ncu 看 kernel 内部

```bash
ncu --set roofline -k regex:"my_kernel" python3 my_script.py
```

关注:
- Achieved Occupancy(目标 >50%)
- Compute / Memory bound 哪个先到达 roofline
- L1 / L2 hit rate
- SMEM 占用是否限制了 occupancy

### 9.4 调 tile 参数

HPC-Ops 算子内部的 tile shape(`kTileM, kTileN, kTileK, kStage`)是编译期参数。如果你想试不同组合,需要:

1. 在 `src/<op>/config.h` 加新的 config 类
2. 在 `src/<op>/<op>.cu` 的 dispatch 函数加新 case
3. 重新编译

这是"编译期特化"的代价:灵活但慢迭代。如果你只是想做参数 sweep 评估性能,推荐先写一个 Python 脚本套 nsys/ncu,而不是把所有组合都编进去。

## 10. 进阶资源

- [CUTLASS / CuTe 官方教程](https://github.com/NVIDIA/cutlass/tree/main/media/docs)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [PTX ISA(WGMMA / TMA / Multicast 详尽规范)](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [PyTorch CustomOp / torch.ops 文档](https://docs.pytorch.org/docs/stable/library.html)

---

下一篇:[11_glossary.md](./11_glossary.md) — 术语表,所有缩写一站式查询。
