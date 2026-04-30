# pt2so FactorMLP TorchScript Runtime

这个工程现在的主链路是：上游用 `torch.jit.trace` 生成 TorchScript `.pt`，本项目把这个 `.pt` 加密后通过 `.incbin` 编译进 `libmodel.so`。运行时外部程序只需要 `dlopen` 这个 `.so`，设置独立保存的 AES key，然后调用 `model_create()` / `model_forward()`。

`.so` 不依赖外部 `.pt`、`.enc` 路径。key 不编译进 `.so`。

## Runtime Shape

输入：

```text
float32 [B, 128]
```

输出：

```text
float32 [B, 1]
```

当前示例模型仍然是 FactorMLP，但 C++ runtime 不再手写这个网络结构，也不再解析 `state_dict`。真正执行的是 TorchScript module：

```cpp
torch::jit::load(...)
module.forward(...)
```

## Build Flow

构建期流程：

```text
Python FactorMLP
  -> torch.jit.trace(...)
  -> artifacts/factor_mlp/factor_mlp.pt
  -> AES-256-GCM 加密成 artifacts/factor_mlp/model.pt.enc
  -> 生成 artifacts/factor_mlp/model.key
  -> tools/generate_blob.py 生成 runtime/src/blob.cpp
  -> .incbin 把 model.pt.enc 链进 libmodel.so
```

运行期流程：

```text
dlopen("libmodel.so")
  -> model_create()
  -> 从 .rodata 读取内嵌密文
  -> 从 PT2SO_MODEL_KEY_FILE 或 PT2SO_MODEL_KEY_HEX 读取 key
  -> 内存中 AES-GCM 解密 TorchScript .pt
  -> torch::jit::load 从内存加载 module
  -> model_forward() 调 module.forward()
```

## Linux Setup

安装系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libssl-dev python3 python3-venv
```

安装 Python 依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

准备 CPU LibTorch，并设置环境变量，例如：

```bash
export Torch_DIR="$PWD/external/libtorch/share/cmake/Torch"
export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:${LD_LIBRARY_PATH:-}"
```

## One Command Pipeline

```bash
source .venv/bin/activate
chmod +x scripts/*.sh
./scripts/run_pipeline_linux.sh
```

成功后会生成：

```text
artifacts/factor_mlp/factor_mlp.pt
artifacts/factor_mlp/sample_input.npy
artifacts/factor_mlp/expected_output.npy
artifacts/factor_mlp/model.pt.enc
artifacts/factor_mlp/model.key
runtime/src/blob.cpp
build/libmodel.so
```

验证通过时会看到：

```text
max_abs_error=...
max_rel_error=...
```

默认要求 `max_abs_error <= 1e-5`。

## Manual Steps

如果你想手动分步跑：

```bash
source .venv/bin/activate

python tools/create_factor_mlp.py
python tools/encrypt_model.py
python tools/generate_blob.py

chmod +x scripts/*.sh
./scripts/build_linux.sh

export PT2SO_MODEL_KEY_FILE="$PWD/artifacts/factor_mlp/model.key"
python tools/validate_ctypes.py
```

如果你想生成更深、更大的测试模型，可以传入隐藏层数量和每层宽度：

```bash
python tools/create_factor_mlp.py --depth 8 --hidden-dim 1024
python tools/encrypt_model.py
python tools/generate_blob.py
./scripts/build_linux.sh
```

`--depth 8` 表示 8 个隐藏层块，每个块是 `Linear -> LayerNorm -> ReLU -> Dropout`。`--hidden-dim 1024` 表示每个隐藏层宽度是 1024。输入仍然是 `[B, 128]`，输出仍然是 `[B, 1]`，所以 C++ runtime 不需要改。

也可以用 hex 字符串提供 key：

```bash
export PT2SO_MODEL_KEY_HEX="$(xxd -p -c 256 artifacts/factor_mlp/model.key)"
```

兼容旧名字：runtime 里也临时支持 `PT2SO_WEIGHTS_KEY_FILE` 和 `PT2SO_WEIGHTS_KEY_HEX`，但新代码建议统一使用 `PT2SO_MODEL_*`。

## incbin

`tools/generate_blob.py` 会生成 `runtime/src/blob.cpp`，核心是 GNU assembler 的 `.incbin`：

```asm
.incbin "/absolute/path/to/artifacts/factor_mlp/model.pt.enc"
```

这表示链接 `libmodel.so` 时，把加密后的 TorchScript 文件原样放进 `.so` 的只读数据段。链接完成后，运行时不再需要 `factor_mlp.pt` 或 `model.pt.enc` 的文件路径。

注意：只把 `model.pt.enc` 编译进 `.so`，不会把 `model.key` 编译进去。

## C ABI

导出的接口在 `runtime/include/model_api.h`：

```c
int model_create(ModelHandle* out_handle);

int model_forward(
    ModelHandle handle,
    const TensorDesc* inputs,
    int input_count,
    TensorDesc* outputs,
    int output_count);

void model_destroy(ModelHandle handle);
const char* model_last_error(ModelHandle handle);
const char* model_version();
```

`model_forward()` 目前只支持：

```text
1 个输入，float32 [B, 128]
1 个输出，float32 [B, 1]
CPU inference
```

## dlopen Test

项目里有一套 Python vs C++ dlopen 对比工具：

```bash
python tools/run_python_pt_reference.py

g++ -std=c++17 -O2 -Iruntime/include tools/run_dlopen_model.cpp -ldl -o build/run_dlopen_model

export PT2SO_MODEL_KEY_FILE=artifacts/factor_mlp/model.key
./build/run_dlopen_model --lib build/libmodel.so

python tools/compare_reference_outputs.py
```

输出文件：

```text
artifacts/factor_mlp/python_pt_output.npy
artifacts/factor_mlp/cpp_dlopen_output.npy
artifacts/factor_mlp/python_vs_cpp_report.txt
```

## Deployment

部署运行时至少需要：

```text
libmodel.so
model.key
```

不需要部署：

```text
factor_mlp.pt
model.pt.enc
sample_input.npy
expected_output.npy
runtime/src/blob.cpp
```

示例：

```bash
export LD_LIBRARY_PATH="/path/to/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PT2SO_MODEL_KEY_FILE=/secure/path/model.key
```

然后你的业务程序再 `dlopen("/path/to/libmodel.so")`。

## Security Notes

`libmodel.so` 里只有密文，没有 key。请不要把 `model.key` 提交到 Git，也不要放进公开镜像或公开下载路径。

当前 `.gitignore` 已经排除：

```text
artifacts/factor_mlp/*.pt
artifacts/factor_mlp/*.npy
artifacts/factor_mlp/*.bin
artifacts/factor_mlp/*.enc
artifacts/factor_mlp/*.key
runtime/src/blob.cpp
build/
*.so
```

## Troubleshooting

如果脚本提示 `Permission denied`：

```bash
chmod +x scripts/*.sh
```

如果 `model_create()` 提示 key 缺失：

```bash
echo "$PT2SO_MODEL_KEY_FILE"
echo "$PT2SO_MODEL_KEY_HEX"
```

如果解密失败，通常是当前 `model.key` 和编译进 `.so` 的 `model.pt.enc` 不是同一轮生成的。

如果加载 `.so` 失败，检查：

```bash
export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:${LD_LIBRARY_PATH:-}"
```

如果 CMake 找不到 LibTorch，检查：

```bash
echo "$Torch_DIR"
```
