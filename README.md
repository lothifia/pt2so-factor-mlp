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
  -> AES-256-GCM weights.enc + weights.key
  -> runtime/src/blob.cpp with GNU assembler .incbin
  -> libmodel.so
```

At runtime:

```text
dlopen("libmodel.so")
  -> model_create()
  -> read embedded encrypted pack from .rodata
  -> decrypt embedded blob in memory
  -> parse weights.bin format from memory
  -> strict-load C++ FactorMLP state_dict
```

The AES key is embedded into `libmodel.so` so the library can be moved anywhere and still create the model without file paths. This is useful packaging and obfuscation, but it is not strong protection against someone who can reverse engineer the shared library.

`tools/generate_blob.py` does not convert binary data into a giant C++ array. It generates a small `runtime/src/blob.cpp` that uses GNU-style inline assembly:

```asm
.incbin "/absolute/path/to/artifacts/factor_mlp/weights.enc"
.incbin "/absolute/path/to/artifacts/factor_mlp/weights.key"
```

Those bytes are linked into the shared object's read-only data section. After linking, `libmodel.so` no longer needs the source `.pt`, `weights.bin`, `weights.enc`, or `weights.key` files at runtime.

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

`model_create()` does not read any external weight path. It only uses encrypted data compiled into the shared library.
