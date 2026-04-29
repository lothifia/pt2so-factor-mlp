# pt2so FactorMLP Embedded Runtime

This project builds a Linux shared library for a fixed FactorMLP architecture. The upstream `.pt` file is used only at build time to extract a PyTorch `state_dict`.

The runtime does not use TorchScript, `torch.jit`, ONNX, graph export, or `torch::jit::load`.

## Runtime Shape

Input:

```text
float32 [B, 128]
```

Output:

```text
float32 [B, 1]
```

Architecture:

```text
Linear(128 -> 256)
LayerNorm(256)
ReLU
Dropout(0.1)
Linear(256 -> 64)
LayerNorm(64)
ReLU
Dropout(0.1)
Linear(64 -> 1)
```

`model_create()` calls `eval()`, so dropout is disabled during inference.

## Embedded Weights Flow

The shared library has no runtime dependency on external `.pt`, `.bin`, or `.enc` paths. The build pipeline is:

```text
factor_mlp.pt
  -> weights.bin
  -> AES-256-GCM weights.enc + external weights.key
  -> runtime/src/blob.cpp with GNU assembler .incbin
  -> libmodel.so
```

At runtime:

```text
dlopen("libmodel.so")
  -> model_create()
  -> read embedded encrypted pack from .rodata
  -> read AES key from PT2SO_WEIGHTS_KEY_HEX or PT2SO_WEIGHTS_KEY_FILE
  -> decrypt embedded blob in memory
  -> parse weights.bin format from memory
  -> strict-load C++ FactorMLP state_dict
```

Only `weights.enc` is embedded into `libmodel.so`. The AES key is not compiled into the shared library. The caller must provide the key before calling `model_create()`:

```bash
export PT2SO_WEIGHTS_KEY_FILE=/secure/path/weights.key
```

or:

```bash
export PT2SO_WEIGHTS_KEY_HEX="$(xxd -p -c 256 /secure/path/weights.key)"
```

`PT2SO_WEIGHTS_KEY_FILE` may point to a raw 32-byte key file or a text file containing the 64-character hex key.

`tools/generate_blob.py` does not convert binary data into a giant C++ array. It generates a small `runtime/src/blob.cpp` that uses GNU-style inline assembly:

```asm
.incbin "/absolute/path/to/artifacts/factor_mlp/weights.enc"
```

The encrypted bytes are linked into the shared object's read-only data section. After linking, `libmodel.so` no longer needs the source `.pt`, `weights.bin`, or `weights.enc` files at runtime. It still needs the AES key supplied by environment variable or key file.

## Linux Build

Install Linux build dependencies, including OpenSSL headers:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libssl-dev python3 python3-venv
```

Install Python requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Download CPU LibTorch and set `Torch_DIR`, for example:

```bash
export Torch_DIR="$PWD/external/libtorch/share/cmake/Torch"
export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:${LD_LIBRARY_PATH:-}"
```

Run the full pipeline:

```bash
chmod +x scripts/*.sh
./scripts/run_pipeline_linux.sh
```

Expected outputs:

```text
artifacts/factor_mlp/factor_mlp.pt
artifacts/factor_mlp/sample_input.npy
artifacts/factor_mlp/expected_output.npy
artifacts/factor_mlp/weights.bin
artifacts/factor_mlp/weights.enc
artifacts/factor_mlp/weights.key
runtime/src/blob.cpp
build/libmodel.so
```

Validation passes when:

```text
max_abs_error <= 1e-5
```

## Manual Build

```bash
source .venv/bin/activate

python tools/create_factor_mlp.py
python tools/extract_weights.py
python tools/encrypt_weights.py
python tools/generate_blob.py

./scripts/build_linux.sh
export PT2SO_WEIGHTS_KEY_FILE="$PWD/artifacts/factor_mlp/weights.key"
python tools/validate_ctypes.py
```

## C ABI

The shared library exports:

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

`model_create()` does not read any external weight path. It only uses encrypted data compiled into the shared library, plus the AES key supplied through `PT2SO_WEIGHTS_KEY_HEX` or `PT2SO_WEIGHTS_KEY_FILE`.

## 中文使用流程

这一节按真实使用顺序说明整个工程怎么跑。当前工程目标是：构建一个 Linux `libmodel.so`，其中只内嵌加密后的模型权重密文，AES key 不编译进 `.so`，而是在运行时通过环境变量或 key 文件提供。

### 1. 准备系统环境

在 Linux 机器上安装基础编译工具、CMake、OpenSSL 头文件和 Python venv：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libssl-dev python3 python3-venv
```

准备 Python 虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

准备 CPU 版 LibTorch，并设置 `Torch_DIR`。例如 LibTorch 解压到了 `external/libtorch`：

```bash
export Torch_DIR="$PWD/external/libtorch/share/cmake/Torch"
export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:${LD_LIBRARY_PATH:-}"
```

### 2. 一键构建和验证

最简单的方式是直接执行：

```bash
chmod +x scripts/*.sh
./scripts/run_pipeline_linux.sh
```

这个脚本会依次执行：

```text
1. 生成 Python FactorMLP checkpoint 和测试样本
2. 从 .pt 提取 state_dict 到 weights.bin
3. 生成 AES-256-GCM key，并把 weights.bin 加密成 weights.enc
4. 生成使用 .incbin 的 runtime/src/blob.cpp
5. 编译 libmodel.so
6. 设置 PT2SO_WEIGHTS_KEY_FILE
7. 用 ctypes 调用 libmodel.so 验证输出
```

如果验证成功，会看到类似：

```text
max_abs_error=...
max_rel_error=...
```

其中 `max_abs_error` 需要小于或等于 `1e-5`。

### 3. 每一步实际生成什么

执行 pipeline 后会生成这些构建期产物：

```text
artifacts/factor_mlp/factor_mlp.pt
artifacts/factor_mlp/sample_input.npy
artifacts/factor_mlp/expected_output.npy
artifacts/factor_mlp/weights.bin
artifacts/factor_mlp/weights.enc
artifacts/factor_mlp/weights.key
runtime/src/blob.cpp
build/libmodel.so
```

各文件含义：

```text
factor_mlp.pt        Python 侧 checkpoint，只在构建期使用
weights.bin          从 state_dict 提取出的自定义明文权重包
weights.enc          AES-256-GCM 加密后的权重包，会被编译进 libmodel.so
weights.key          AES-256 key，不会被编译进 libmodel.so，需要单独安全保存
runtime/src/blob.cpp 自动生成的 incbin 源文件，用于把 weights.enc 链进 .so
libmodel.so          最终部署用动态库
```

### 4. 手动分步执行

如果不想跑一键脚本，也可以手动执行：

```bash
source .venv/bin/activate

python tools/create_factor_mlp.py
python tools/extract_weights.py
python tools/encrypt_weights.py
python tools/generate_blob.py

./scripts/build_linux.sh
```

验证时需要先提供 key：

```bash
export PT2SO_WEIGHTS_KEY_FILE="$PWD/artifacts/factor_mlp/weights.key"
python tools/validate_ctypes.py
```

### 5. incbin 是怎么工作的

`tools/generate_blob.py` 会生成 `runtime/src/blob.cpp`，里面使用 GNU assembler `.incbin`：

```asm
.incbin "/absolute/path/to/artifacts/factor_mlp/weights.enc"
```

这表示编译/链接时，把 `weights.enc` 的二进制内容原样放进 `libmodel.so` 的只读数据区。链接完成后，运行时不再需要 `weights.enc` 文件路径。

注意：当前只 `.incbin` 加密后的 `weights.enc`，不会 `.incbin` `weights.key`。

### 6. 运行时如何提供 key

`model_create()` 会在运行时读取 key。支持两种方式，优先级如下：

```text
1. PT2SO_WEIGHTS_KEY_HEX
2. PT2SO_WEIGHTS_KEY_FILE
```

方式一：使用 key 文件：

```bash
export PT2SO_WEIGHTS_KEY_FILE=/secure/path/weights.key
```

方式二：使用 hex 字符串：

```bash
export PT2SO_WEIGHTS_KEY_HEX="$(xxd -p -c 256 /secure/path/weights.key)"
```

`PT2SO_WEIGHTS_KEY_FILE` 可以指向原始 32 字节 key 文件，也可以指向包含 64 个十六进制字符的文本文件。

如果没有设置 key，`model_create()` 会失败，并且可以通过 `model_last_error(nullptr)` 或对应 handle 的 `model_last_error(handle)` 查看错误。

### 7. 部署时需要带哪些文件

部署运行时至少需要：

```text
build/libmodel.so
weights.key
```

不需要带：

```text
factor_mlp.pt
weights.bin
weights.enc
sample_input.npy
expected_output.npy
runtime/src/blob.cpp
```

部署时推荐把 key 放在安全路径，例如：

```bash
/etc/pt2so/weights.key
```

然后启动服务前设置：

```bash
export PT2SO_WEIGHTS_KEY_FILE=/etc/pt2so/weights.key
```

### 8. 外部程序调用顺序

外部 C/C++/Python/Rust/Go 程序通过 `dlopen` 加载 `libmodel.so` 后，调用顺序是：

```text
1. 设置 PT2SO_WEIGHTS_KEY_FILE 或 PT2SO_WEIGHTS_KEY_HEX
2. dlopen("libmodel.so")
3. model_create(&handle)
4. 准备输入 TensorDesc，shape 为 [B, 128]，dtype 为 MODEL_FLOAT32
5. 第一次 model_forward 可以用空 output buffer 探测输出大小
6. 分配 output buffer
7. 第二次 model_forward 得到结果
8. model_destroy(handle)
```

输出 shape 固定是：

```text
[B, 1]
```

### 9. 安全注意事项

`libmodel.so` 里只有密文，没有 key。这样比把 key 和密文一起编译进 `.so` 更合理。

仍然需要注意：

```text
weights.key 不要提交到 Git
weights.key 不要打进公开镜像
weights.key 不要和 libmodel.so 放在同一个公开下载位置
生产环境用权限控制保护 key 文件
```

当前 `.gitignore` 已经排除了：

```text
artifacts/factor_mlp/*.key
artifacts/factor_mlp/*.enc
artifacts/factor_mlp/*.bin
artifacts/factor_mlp/*.pt
runtime/src/blob.cpp
build/
*.so
```

### 10. 常见问题

如果 `model_create()` 报 key 缺失，检查：

```bash
echo "$PT2SO_WEIGHTS_KEY_FILE"
echo "$PT2SO_WEIGHTS_KEY_HEX"
```

如果报解密失败，通常是：

```text
libmodel.so 内嵌的 weights.enc 和当前 weights.key 不匹配
key 文件被损坏
PT2SO_WEIGHTS_KEY_HEX 内容不是正确的 64 字符十六进制字符串
```

如果 `ctypes` 加载 `libmodel.so` 失败，检查：

```bash
export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:${LD_LIBRARY_PATH:-}"
```

如果 CMake 找不到 LibTorch，检查：

```bash
echo "$Torch_DIR"
```

它应该指向：

```text
external/libtorch/share/cmake/Torch
```
