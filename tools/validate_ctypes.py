from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "factor_mlp"

MODEL_FLOAT32 = 1


class TensorDesc(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("dtype", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int64 * 8),
        ("bytes", ctypes.c_size_t),
    ]


def candidate_library_paths() -> list[Path]:
    return [
        ROOT / "build" / "libmodel.so",
        ROOT / "build" / "Release" / "libmodel.so",
    ]


def find_library() -> Path:
    for path in candidate_library_paths():
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in candidate_library_paths())
    raise FileNotFoundError(f"compiled shared library not found; tried:\n{tried}")


def make_desc(array: np.ndarray) -> TensorDesc:
    array = np.ascontiguousarray(array)
    desc = TensorDesc()
    desc.data = ctypes.c_void_p(array.ctypes.data)
    desc.dtype = MODEL_FLOAT32
    desc.ndim = array.ndim
    for i in range(8):
        desc.shape[i] = array.shape[i] if i < array.ndim else 0
    desc.bytes = array.nbytes
    return desc


def last_error(lib: ctypes.CDLL, handle: ctypes.c_void_p | None = None) -> str:
    raw = lib.model_last_error(handle)
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def main() -> None:
    sample_input = np.load(ARTIFACT_DIR / "sample_input.npy").astype(np.float32, copy=False)
    expected_output = np.load(ARTIFACT_DIR / "expected_output.npy").astype(np.float32, copy=False)

    lib = ctypes.CDLL(str(find_library()))

    lib.model_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.model_create.restype = ctypes.c_int
    lib.model_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(TensorDesc),
        ctypes.c_int,
        ctypes.POINTER(TensorDesc),
        ctypes.c_int,
    ]
    lib.model_forward.restype = ctypes.c_int
    lib.model_destroy.argtypes = [ctypes.c_void_p]
    lib.model_destroy.restype = None
    lib.model_last_error.argtypes = [ctypes.c_void_p]
    lib.model_last_error.restype = ctypes.c_char_p

    handle = ctypes.c_void_p()
    rc = lib.model_create(ctypes.byref(handle))
    if rc != 0:
        raise RuntimeError(f"model_create failed rc={rc}: {last_error(lib)}")

    try:
        input_desc = make_desc(sample_input)
        output_probe = TensorDesc()
        rc = lib.model_forward(
            handle,
            ctypes.byref(input_desc),
            1,
            ctypes.byref(output_probe),
            1,
        )
        if rc != -3:
            raise RuntimeError(f"expected output-size probe to return -3, got {rc}: {last_error(lib, handle)}")

        output_shape = tuple(output_probe.shape[i] for i in range(output_probe.ndim))
        output = np.empty(output_shape, dtype=np.float32)
        output_desc = make_desc(output)

        rc = lib.model_forward(
            handle,
            ctypes.byref(input_desc),
            1,
            ctypes.byref(output_desc),
            1,
        )
        if rc != 0:
            raise RuntimeError(f"model_forward failed rc={rc}: {last_error(lib, handle)}")

        if output.shape != expected_output.shape:
            raise AssertionError(f"shape mismatch: got {output.shape}, expected {expected_output.shape}")

        abs_error = np.abs(output - expected_output)
        rel_error = abs_error / np.maximum(np.abs(expected_output), 1e-12)
        max_abs_error = float(abs_error.max())
        max_rel_error = float(rel_error.max())

        print(f"max_abs_error={max_abs_error:.9g}")
        print(f"max_rel_error={max_rel_error:.9g}")

        if max_abs_error > 1e-5:
            raise AssertionError(f"max_abs_error exceeds threshold: {max_abs_error}")
    finally:
        lib.model_destroy(handle)


if __name__ == "__main__":
    main()
