#pragma once

#include <stddef.h>
#include <stdint.h>

#define MODEL_API_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

typedef void* ModelHandle;

enum ModelDType {
    MODEL_FLOAT32 = 1,
    MODEL_FLOAT16 = 2,
    MODEL_FLOAT64 = 3,
    MODEL_INT64 = 4,
    MODEL_INT32 = 5,
    MODEL_UINT8 = 6,
    MODEL_BOOL = 7
};

struct TensorDesc {
    void* data;
    int dtype;
    int ndim;
    int64_t shape[8];
    size_t bytes;
};

struct ModelCreateTiming {
    const char* algorithm;
    size_t encrypted_bytes;
    size_t plaintext_bytes;
    double decrypt_ms;
    double jit_load_ms;
    double total_ms;
};

MODEL_API_EXPORT int model_create(ModelHandle* out_handle);

MODEL_API_EXPORT int model_forward(
    ModelHandle handle,
    const TensorDesc* inputs,
    int input_count,
    TensorDesc* outputs,
    int output_count);

MODEL_API_EXPORT void model_destroy(ModelHandle handle);

MODEL_API_EXPORT const char* model_last_error(ModelHandle handle);

MODEL_API_EXPORT int model_last_create_timing(ModelHandle handle, struct ModelCreateTiming* out_timing);

MODEL_API_EXPORT const char* model_version();

#ifdef __cplusplus
}
#endif
