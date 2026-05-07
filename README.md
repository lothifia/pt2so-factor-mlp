# pt2so FactorMLP TorchScript Runtime

这个工程把上游 `torch.jit.trace` 导出的 TorchScript `.pt` 加密后嵌入进 `libmodel.so`。运行时业务程序 `dlopen` 这个 `.so` 后直接调用：

```text
model_create()
model_forward()
model_destroy()
```

当前版本：

```text
AES-128-GCM
OpenSSL libcrypto.a 静态链接进 libmodel.so
AES key 混淆后编译进 libmodel.so
```

部署环境不需要安装 `libssl-dev`，不需要 `libcrypto.so`，不需要 `libssl.so`，也不需要 `AF_ALG` 权限。

注意：key 被混淆后内嵌在 `.so` 里，不会以明显的 `static uint8_t key[16]` 形式出现；但这仍然不是严格意义上的不可逆白盒安全。拿到 `.so` 的强逆向者仍可能通过调试或 dump 内存恢复模型或 key。

## Runtime Shape

```text
input:  float32 [B, 128]
output: float32 [B, 1]
```

`B` 是动态 batch size，`128` 是固定特征数。

## Model

默认 FactorMLP：

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

导出方式：

```text
torch.jit.trace(...)
torch.jit.freeze(...)
traced.save("factor_mlp.pt")
```

## Build Flow

加密嵌入版本：

```text
factor_mlp.pt
  -> tools/encrypt_model.py
     1. 生成随机 AES-128 key K
     2. 用 K 做 AES-128-GCM 加密，得到 weights.enc
     3. 把 K 拆成 key shares
     4. 每个 share 做 rotate / xor / add / reorder 编码
     5. 生成 runtime/src/aes_key_obfuscated.cpp
  -> tools/generate_blob.py
     用 .incbin 把 weights.enc 嵌入 runtime/src/blob.cpp
  -> scripts/build_linux.sh
     静态链接 libcrypto.a，编译 libmodel.so
```

运行阶段：

```text
model_create()
  -> 读取 embedded weights.enc
  -> 从混淆 key shares 临时恢复 AES-128 key
  -> OpenSSL EVP AES-128-GCM 解密并认证
  -> torch::jit::load 从内存加载模型
  -> 清理临时 key 和明文 buffer
```

明文 baseline：

```text
factor_mlp.pt
  -> tools/generate_blob.py --plaintext
  -> runtime/src/blob.cpp 使用 .incbin 嵌入明文 .pt
  -> libmodel.so
```

明文 baseline 也是把 `.pt` 编译进 `.so`，不是运行时从磁盘路径读取 `.pt`。

## Linux Setup

构建环境需要 OpenSSL 头文件和 `libcrypto.a`。Ubuntu/Debian 上通常：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libssl-dev python3 python3-venv binutils
```

Python 环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

LibTorch：

```bash
export Torch_DIR="$PWD/external/libtorch/share/cmake/Torch"
export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:${LD_LIBRARY_PATH:-}"
```

## Static OpenSSL

默认构建：

```text
STATIC_OPENSSL=ON
```

也就是把 `libcrypto.a` 静态链接进 `libmodel.so`。部署环境不需要：

```text
libssl-dev
libcrypto.so
libssl.so
```

如果 CMake 没自动找到静态库，可以显式指定：

```bash
LIBCRYPTO_A=/path/to/libcrypto.a \
OPENSSL_INCLUDE_DIR=/path/to/openssl/include \
./scripts/build_linux.sh
```

`libcrypto.a` 需要能被链接进 shared library。如果链接时报 relocation / `-fPIC` 相关错误，需要换用带 PIC 编译的 OpenSSL 静态库。

构建脚本会自动调用：

```bash
scripts/check_runtime_deps_linux.sh build/libmodel.so
```

如果 `readelf -d libmodel.so` 里出现 `libcrypto.so` 或 `libssl.so`，脚本会失败。

## One Command Pipeline

第一次在 Linux 上如果脚本没有执行权限，先运行：

```bash
chmod +x scripts/*.sh
```

完整流程：

```bash
source .venv/bin/activate
./scripts/run_pipeline_linux.sh
```

默认生成：

```text
artifacts/factor_mlp/factor_mlp.pt
artifacts/factor_mlp/sample_input.npy
artifacts/factor_mlp/expected_output.npy
artifacts/factor_mlp/weights.enc
runtime/src/blob.cpp
runtime/src/aes_key_obfuscated.cpp
build/libmodel.so
```

`runtime/src/aes_key_obfuscated.cpp` 含有混淆后的 key material，是生成产物，已经被 `.gitignore` 忽略，不建议提交。

## Manual Commands

也可以拆开跑：

```bash
python tools/create_factor_mlp.py --output-dir artifacts/factor_mlp

python tools/encrypt_model.py \
  --input artifacts/factor_mlp/factor_mlp.pt \
  --output artifacts/factor_mlp/weights.enc \
  --key-source runtime/src/aes_key_obfuscated.cpp

python tools/generate_blob.py \
  --encrypted artifacts/factor_mlp/weights.enc \
  --output runtime/src/blob.cpp

./scripts/build_linux.sh
python tools/validate_ctypes.py
```

调整 key share 数量：

```bash
python tools/encrypt_model.py \
  --input artifacts/factor_mlp/factor_mlp.pt \
  --output artifacts/factor_mlp/weights.enc \
  --key-source runtime/src/aes_key_obfuscated.cpp \
  --key-share-count 8
```

## Benchmark

比较“明文嵌入直接 load”和“AES-128-GCM 解密后 load”：

```bash
REPEAT=10 WARMUP=2 ./scripts/benchmark_crypto_linux.sh
```

输出：

```text
artifacts/crypto_bench/results.csv
```

字段含义：

```text
algorithm                  当前 case 名称，plaintext-embedded 或 gcm-aes-128
iteration                  第几次正式计时，不包含 warmup
encrypted_bytes            嵌入的 AES-GCM 包大小；明文 baseline 为 0
plaintext_bytes            解密后 TorchScript .pt 字节数
decrypt_ms                 C++ 中恢复混淆 key 并做 AES-128-GCM 解密/认证耗时
jit_load_ms                torch::jit::load 从内存加载 TorchScript 的耗时
total_create_inner_ms      model_create() 内部总耗时
total_create_wall_ms       benchmark 进程在外部测得的 create 调用墙钟耗时
```

## Bigger Model

增大模型深度和宽度：

```bash
MODEL_ARGS="--depth 16 --hidden-dim 2048" REPEAT=10 WARMUP=2 ./scripts/benchmark_crypto_linux.sh
```

增大样例 batch：

```bash
python tools/create_factor_mlp.py --batch-size 4096 --output-dir artifacts/factor_mlp
```

输入仍然是 `[B, 128]`，输出仍然是 `[B, 1]`。调大 `depth`、`hidden-dim` 会增大模型文件，适合放大加载和解密时间差异。

## Deployment

部署运行时至少需要：

```text
libmodel.so
LibTorch runtime libraries
```

不需要部署：

```text
factor_mlp.pt
weights.enc
runtime/src/blob.cpp
runtime/src/aes_key_obfuscated.cpp
libcrypto.so
libssl.so
```
