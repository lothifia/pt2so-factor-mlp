#pragma once

#include "model_api.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pt2so {

struct ModelSessionCreateTiming {
    std::string algorithm;
    size_t encrypted_bytes{0};
    size_t plaintext_bytes{0};
    double decrypt_ms{0.0};
    double jit_load_ms{0.0};
    double total_ms{0.0};
};

struct ModelOutput {
    std::vector<float> data;
    int64_t rows{0};
    int64_t cols{0};
};

class ModelSession {
public:
    ModelSession();
    ~ModelSession();

    ModelSession(const ModelSession&) = delete;
    ModelSession& operator=(const ModelSession&) = delete;
    ModelSession(ModelSession&&) noexcept;
    ModelSession& operator=(ModelSession&&) noexcept;

    ModelOutput forward(const float* input, int64_t rows, int64_t features);
    ModelOutput forward(const std::vector<float>& input, int64_t rows, int64_t features);

    int forward_c(
        const TensorDesc* inputs,
        int input_count,
        TensorDesc* outputs,
        int output_count) noexcept;

    const ModelSessionCreateTiming& create_timing() const noexcept;
    const std::string& last_error() const noexcept;
    const char* last_error_c_str() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pt2so
