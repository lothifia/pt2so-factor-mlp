#pragma once

#include <cstdint>

#include <torch/torch.h>

struct FactorMLPImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    torch::nn::LayerNorm ln1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::LayerNorm ln2{nullptr};
    torch::nn::Linear fc3{nullptr};
    torch::nn::Dropout dropout{nullptr};

    FactorMLPImpl(
        int64_t num_features,
        int64_t hidden1,
        int64_t hidden2,
        int64_t output_dim,
        double dropout_p);

    torch::Tensor forward(torch::Tensor x);

private:
    int64_t num_features_;
};

TORCH_MODULE(FactorMLP);
