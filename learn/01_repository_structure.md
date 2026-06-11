# 01 - 仓库结构与构建系统

> 本文目标:让你能在 30 分钟内**完整理解一个 `import hpc; hpc.attention_decode_bf16(...)` 调用是如何从 Python 走到 CUDA kernel 的**,并能在本地编译、运行测试。

## 1. 顶层目录速览

```
hpc-ops/
├── README.md              ← 项目主页(性能、Quick Start)
├── LICENSE.txt            ← MIT
├── Makefile               ← 一键 build / test / format
├── setup.py               ← Python 包安装入口(委托给 CMake)
├── CMakeLists.txt         ← C++/CUDA 构建配置
├── conftest.py            ← pytest 配置(含 compute-sanitizer hook)
├── requirements-dev.txt   ← 开发依赖(torch==2.7.0, numpy, pytest 等)
├── CPPLINT.cfg            ← cpplint 配置
├── .clang-format          ← C++ 格式化规则
│
├── hpc/                   ← **Python 接口层**(用户 import 的入口)
│   ├── __init__.py        ←   动态发现并导出所有 .py 文件里的函数
│   ├── attention.py       ←   attention_prefill_bf16 等
│   ├── gemm.py            ←   gemm_bf16xfp32
│   ├── group_gemm.py
│   ├── fuse_moe.py
│   ├── allreduce.py
│   ├── sampler.py
│   ├── normalization.py
│   ├── act.py
│   ├── rope.py
│   ├── stem.py
│   ├── communicator.py     ←   MulticastCommunicator 的 Python 类
│   ├── multicast_handle.py ←   多机 multicast buffer 管理
│   └── (build 时生成 _C.abi3.so 和 version.py)
│
├── src/                   ← **C++/CUDA 实现层**
│   ├── C/                 ←   torch::library 注册总入口(stub)
│   ├── attention/
│   │   ├── entry.cc       ←     C++ 入口(每个算子家族都有)
│   │   ├── prefill/       ←     prefill 实现
│   │   └── decode/        ←     decode 实现 + 任务调度
│   ├── gemm/sm90/
│   ├── group_gemm/
│   │   └── cp_async/      ←     低延迟 cp.async 路径
│   ├── fuse_moe/
│   │   └── cp_async/
│   ├── allreduce/
│   │   ├── fuse_allreduce_rmsnorm_high_throughput.cu
│   │   └── fuse_allreduce_rmsnorm_low_latency.cu
│   ├── sampler/
│   ├── normalization/
│   ├── activation/
│   ├── rope/
│   ├── stem/
│   ├── communicator/      ←   单机 NVLink multicast 通信器(纯 C++)
│   └── utils/             ←   公共工具(TMA descriptor、SM count、向量类型)
│
├── 3rd/
│   └── cutlass/include/   ← CUTLASS 源码直接 copy 进来(不用 submodule)
│
├── tests/                 ← pytest 测试,每个 .py 对应一个算子
│   ├── test_attention_*.py
│   ├── test_gemm_bf16xfp32.py
│   ├── test_fuse_moe_*.py
│   ├── ...
│   └── utils.py           ← allclose 等公共辅助
│
├── benchmark/             ← 基准测试,带 README
│   ├── attention_decode/
│   ├── fused_moe/
│   ├── fuse_allreduce_rmsorm/
│   ├── route_gemm/
│   └── sampler/
│
└── assets/                ← README 用的 logo 等
```

## 2. 构建系统:一个 `make wheel` 背后发生了什么

### 2.1 工具链需求

按 `README.md`:
- NVIDIA SM90 GPU(H100 / H20 / H200)
- CUDA Toolkit 12.8+
- Python 3.8+
- C++17 编译器
- 通过 `requirements-dev.txt` 安装的 `torch==2.7.0` 等

### 2.2 顶层入口:`Makefile`

```makefile
all:
    python3 setup.py build      # 增量编译

wheel:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    python3 -m build --wheel --no-isolation  # 打 wheel,不创建隔离虚拟环境

test:
    @for test in tests/test_*.py; do python3 -m pytest -v $$test || exit 1; done

sanitizer:
    @rm -rf /dev/shm/tmp_hpc_*
    @for test in tests/test_*.py; do \
      SANITIZER_CHECK=synccheck,memcheck,racecheck \
      python3 -m pytest -v $$test || exit 1; \
    done

clean:
    rm -rf build dist hpc_ops.egg-info hpc.egg-info .pytest_cache ...
```

### 2.3 `setup.py`:Python 包定义

```python
class CMakeExtension(Extension):
    """声明一个 'C 扩展',但实际构建委托给 CMake。"""

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # 1. 在 build_temp 下调用 cmake 配置
        subprocess.check_call(["cmake", ext.sourcedir, ...])
        # 2. cmake --build . -j16  实际编译
        subprocess.check_call(["cmake", "--build", ".", "-j16"])
        # 3. 把生成的 _C.abi3.so 拷到 hpc/ 包目录
        shutil.copy(so_src, "hpc/_C.abi3.so")

def get_version():
    """版本号 = '0.0.1.dev0+g<git-short-hash>'。 """
    git_hash = subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"]).strip()
    return f"0.0.1.dev0+g{git_hash}", git_hash

# 把版本号写进 hpc/version.py
with open("hpc/version.py", "w") as fp:
    fp.write(f'version = "{version}"\n')

setup(
    name="hpc-ops",
    packages=["hpc"],
    ext_modules=[CMakeExtension("hpc", version_macros)],
    cmdclass={"build_ext": CMakeBuild},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},  # abi3 stable ABI
    install_requires=["torch"],
)
```

**关键点**:`py_limited_api: cp39` 让一份 `.so` 适配 Python 3.9+ 所有版本(用 Python 稳定 ABI)。

### 2.4 `CMakeLists.txt`:CUDA 编译

```cmake
cmake_minimum_required(VERSION 3.18)
project(hpc_ops LANGUAGES CXX CUDA)

# 1. 收集所有 .cu / .cc 源文件
file(GLOB_RECURSE SOURCES "src/*/*.cu" "src/*/*.cc")
# 排除测试文件
list(FILTER SOURCES EXCLUDE REGEX ".*test.*")

# 2. 构建一个共享库 _C.abi3.so
add_library(_C MODULE ${SOURCES})

# 3. 关键编译选项
set_target_properties(_C PROPERTIES
  CUDA_ARCHITECTURES "90a"        # 编译 SM90 PTX/SASS(注意 'a' 是 Hopper 完整指令集)
  CXX_STANDARD 17
  CUDA_STANDARD 17
  PREFIX "" SUFFIX ".abi3.so"
)

# 4. 头文件路径
target_include_directories(_C PRIVATE
  ./                              # 源码根目录,这样 src/xxx/yyy.h 都能找到
  3rd/cutlass/include             # CUTLASS
  ${TORCH_INCLUDE_PATHS}          # libtorch 头文件(从 PyTorch 安装目录读)
)

# 5. 链接 libtorch
target_link_libraries(_C PRIVATE
  cuda c10 torch torch_cpu cudart c10_cuda torch_cuda
)

# 6. NVCC 编译选项
target_compile_options(_C PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo                        # 保留行号信息(给 nsys/ncu 看)
    --expt-relaxed-constexpr         # 允许设备代码里写宽松的 constexpr
    -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED   # 启用 CuTe 的 SM90 扩展 MMA 形状
  >
)
```

**注意**:`90a` 不是 `90`!Hopper 的 wgmma、TMA 是 `sm_90a` 独有的指令集扩展。如果误写 `90` 编译会成功但运行时 illegal instruction。

### 2.5 编译产物

```
build/
└── lib.linux-x86_64-cpython-3X/
    └── hpc/
        ├── __init__.py
        ├── attention.py
        ├── ...
        ├── version.py
        └── _C.abi3.so      ← 真正的 CUDA 内核全在这里
```

一旦构建完成,你既可以 `pip install dist/*.whl`,也可以直接 `sys.path.insert(0, "build/lib.*")` 来本地引用(测试脚本里都这么干)。

---

## 3. Python 接口层:`hpc/__init__.py` 的"魔法"

`hpc/__init__.py` 不到 50 行,但巧妙地把整个 Python 接口拼出来:

```python
import torch
import importlib
import os, sys
from pathlib import Path

_pkg_dir = Path(__file__).parent

def _discover_modules() -> dict:
    """扫描 hpc/ 目录下所有 .py(忽略 _ 开头),import 它们。"""
    modules = {}
    for file in _pkg_dir.iterdir():
        if file.suffix != ".py" or file.name.startswith("_"):
            continue
        module_name = file.stem
        module = importlib.import_module(f".{module_name}", package=__package__)
        modules[module_name] = module
    return modules

def _export_functions(modules):
    """把每个子模块里的 callable 提升到 hpc.* 命名空间。"""
    for module_name, module in modules.items():
        funcs = {n: o for n, o in vars(module).items()
                 if callable(o) and not n.startswith("_")}
        globals().update(funcs)
        __all__.extend(funcs.keys())

# 关键 3 行:
so_files = list(Path(__file__).parent.glob("_C.*.so"))
assert len(so_files) == 1
torch.ops.load_library(so_files[0])     # ← 加载 CUDA 共享库

__all__ = []
_export_functions(_discover_modules())  # ← 自动导出所有 Python wrapper

__version__ = torch.ops.hpc.version()           # 从 .so 读版本
__built_json__ = torch.ops.hpc.built_json()     # 从 .so 读构建元数据
```

**结果**:你写 `hpc.attention_decode_bf16(...)`,等价于 `hpc.attention.attention_decode_bf16(...)`。

---

## 4. Python → C++ → CUDA 的完整调用链

以最常用的 `hpc.attention_decode_bf16` 为例,逐层追踪。

### 4.1 Python 层:`hpc/attention.py`

```python
def attention_decode_bf16(
    q, kcache, vcache, block_ids, num_seq_kvcache,
    mtp=0, new_kv_included=False, splitk=True,
    split_flag=None, output=None,
) -> Tensor:
    """完整 docstring 见源码,描述 Q/K/V 张量形状要求"""
    return torch.ops.hpc.attention_decode_bf16(
        q, kcache, vcache, block_ids, num_seq_kvcache,
        mtp, new_kv_included, splitk, split_flag, output,
    )

@torch.library.register_fake("hpc::attention_decode_bf16")
def attention_decode_bf16_fake(q, kcache, vcache, ...):
    """torch.compile 用的形状推断函数。"""
    return torch.empty_like(q)
```

两个东西在这里发生:
1. **真实调用**通过 `torch.ops.hpc.<name>` 路由到 C++ 端(由 `torch.ops.load_library` 注册)。
2. **fake/meta 实现**告诉 `torch.compile` 输出形状,这样模型可以走 `torch.compile` 路径。

### 4.2 C++ 入口:`src/attention/entry.cc`

```cpp
namespace hpc { namespace attention {

torch::Tensor attention_decode_bf16_entry(
    const torch::Tensor &q,
    torch::Tensor &kcache,
    torch::Tensor &vcache,
    const torch::Tensor &block_ids,
    const torch::Tensor &num_seq_kvcache,
    int64_t mtp,
    bool new_kv_included,
    bool use_splitk,
    std::optional<torch::Tensor> split_flag,
    std::optional<torch::Tensor> output)
{
    // 1. 取当前 CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

    // 2. 验证输入(TORCH_CHECK 类似 assert)
    TORCH_CHECK(q.device().is_cuda(), ...);
    TORCH_CHECK(num_dim_qk == 128, "we only support head dim 128.");
    TORCH_CHECK(block_size == 32 || block_size == 64, ...);
    TORCH_CHECK(heads_per_group == 4 || heads_per_group == 8, ...);

    // 3. 推断/分配输出
    auto y = output.has_value() ? output.value() :
             torch::empty({num_batch * num_seq_q, num_head_q, num_dim_v}, options);

    // 4. 决定是否走 split-k
    int splitk = use_splitk ? (num_batch <= 32 ? 16 : 4) : 1;
    if (splitk > 1) {
        // 分配中间张量 lse, split_out, split_flag
        ...
    }

    // 5. 调用 CUDA 内核
    decode::attention_decode_bf16_async(
        y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, ...
        stream);

    return y;
}

}}  // namespace

// 6. 通过 TORCH_LIBRARY_FRAGMENT 注册到 PyTorch 算子库
TORCH_LIBRARY_FRAGMENT(hpc, m) {
    m.def("attention_decode_bf16(Tensor q, Tensor! kcache, Tensor! vcache, ...) -> (Tensor)");
    m.impl("attention_decode_bf16", torch::kCUDA, &attention_decode_bf16_entry);
}
```

**关键细节**:
- `Tensor!` 表示 in-place 修改(让 PyTorch 知道这些张量会被写入)。
- `TORCH_LIBRARY_FRAGMENT(hpc, m)` 让多个 `.cc` 文件都能向同一个 `hpc::` 命名空间注册算子。
- 顶层 `src/C/C.cc` 只有一个空的 `TORCH_LIBRARY(hpc, m) {}` 用来**初始化**命名空间;真正的注册都在各家族的 `entry.cc` 里通过 `TORCH_LIBRARY_FRAGMENT` 增补。

### 4.3 CUDA dispatch:`src/attention/decode/decode.cc`

```cpp
bool attention_decode_bf16_async(
    void *y_ptr, void *lse_ptr, void *split_out_ptr,
    const void *q_ptr, void *kcache_ptr, void *vcache_ptr,
    ...
    cudaStream_t stream)
{
    if (num_dim_qk == 128) {
        return smallm_bf16_dim128_static_async(...);  // 真正的 kernel launcher
    }
    return false;
}
```

这一层是"特征分派":根据 head dim、quant type、SM 版本选具体的 kernel 实现。

### 4.4 CUDA kernel:`src/attention/decode/sm90/static/smallm_bf16_dim128_static.cu`

里面是 CUDA `__global__` 函数,使用 CuTe + WGMMA + TMA 写的 attention 内核。具体实现见 [02_attention.md](./02_attention.md)。

### 4.5 调用链总结

```
hpc.attention_decode_bf16(q, ...)                         ← Python user
    ↓
torch.ops.hpc.attention_decode_bf16(q, ...)               ← torch dispatcher
    ↓
hpc::attention::attention_decode_bf16_entry(q, ...)       ← C++ entry (src/attention/entry.cc)
    ↓ (验证、分配、决定 splitk)
hpc::attention::decode::attention_decode_bf16_async(...)  ← CUDA dispatch (decode.cc)
    ↓ (按 dim 选实现)
hpc::attention::decode::smallm_bf16_dim128_static_async(...)  ← kernel launcher (.cu)
    ↓ <<<grid, block, smem, stream>>>
__global__ smallm_attention_decode_kernel<...>(...)       ← 真正的 CUDA kernel
```

---

## 5. 测试基础设施:`conftest.py`

`conftest.py` 提供一个特别有用的功能 —— **自动 compute-sanitizer 包装**:

```python
# tests/conftest.py 节选
class TraceHook(object):
    def __init__(self, checks, module_name):
        self.checks_ = checks  # 例如 ['memcheck', 'synccheck']
        self.module_name = module_name

    def _wrap_func(self, module, func_name):
        org_func = getattr(module, func_name)

        def wrapped(*args, **kwargs):
            # 1. 把输入张量 torch.save 到临时文件
            save_data(tmp_before, "hpc", func_name, None, args, kwargs)
            # 2. 真正调用,捕获输出
            ret = org_func(*args, **kwargs)
            # 3. 把输出 torch.save
            save_data(tmp_after, "hpc", func_name, ret, args, kwargs)
            # 4. 生成一个独立的 reproducer python 脚本
            dump_test_py(tmp_py, tmp_before, tmp_after, pypath)
            # 5. 对该脚本调用 compute-sanitizer
            for check in self.checks_:
                sanitizer_check(tmp_py, check)
            return ret

        setattr(module, func_name, wrapped)
```

**用法**:

```bash
# 普通测试
make test

# 跑测试 + compute-sanitizer (慢很多,但能查出内存错误、竞争条件、syncthread 错配)
make sanitizer
```

---

## 6. 实战:跑一个 hello world

假设环境已经装好(SM90 GPU + CUDA 12.8 + Python 3.8+ + PyTorch 2.7):

```bash
# 1. 克隆
git clone https://github.com/Tencent/hpc-ops.git
cd hpc-ops

# 2. 安装开发依赖
python3 -m pip install -r requirements-dev.txt

# 3. 编译(需要 ~5-10 分钟)
make all       # 等价 python3 setup.py build

# 4. 跑一个测试看是否成功
python3 -m pytest -v tests/test_attention_decode_bf16.py -k "num_batch1 and num_seq_q1"
```

如果一切正常,你会看到 `passed`。**到这一步,你已经成功跑通了整个 C++/CUDA 工具链。**

### 接下来

- 想了解某个算子细节 → 对应章节 (02-09)
- 想自己写新算子 → [10_build_and_test.md](./10_build_and_test.md)
- 不懂术语 → [11_glossary.md](./11_glossary.md)
- 重新看 GPU 基础 → [00_background.md](./00_background.md)
