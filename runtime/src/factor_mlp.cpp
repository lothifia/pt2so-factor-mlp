#include "factor_mlp.h"

#include <stdexcept>
#include <string>
#include <vector>

FactorMLPImpl::FactorMLPImpl(
    int64_t num_features,
    int64_t hidden1,
    int64_t hidden2,
    int64_t output_dim,
    double dropout_p)
    : fc1(torch::nn::LinearOptions(num_features, hidden1)),
      ln1(torch::nn::LayerNormOptions(std::vector<int64_t>{hidden1})),
      fc2(torch::nn::LinearOptions(hidden1, hidden2)),
      ln2(torch::nn::LayerNormOptions(std::vector<int64_t>{hidden2})),
      fc3(torch::nn::LinearOptions(hidden2, output_dim)),
      dropout(torch::nn::DropoutOptions(dropout_p)),
      num_features_(num_features) {
    register_module("fc1", fc1);
    register_module("ln1", ln1);
    register_module("fc2", fc2);
    register_module("ln2", ln2);
    register_module("fc3", fc3);
    register_module("dropout", dropout);
}

torch::Tensor FactorMLPImpl::forward(torch::Tensor x) {
    if (!x.defined()) {
        throw std::runtime_error("FactorMLP input tensor is undefined");
    }
    if (x.dim() != 2) {
        throw std::runtime_error("FactorMLP expected input rank 2, got rank " + std::to_string(x.dim()));
    }
    if (x.size(1) != num_features_) {
        throw std::runtime_error(
            "FactorMLP expected feature dimension " + std::to_string(num_features_) +
            ", got " + std::to_string(x.size(1)));
    }

    x = fc1->forward(x);
    x = ln1->forward(x);
    x = torch::relu(x);
    x = dropout->forward(x);
    x = fc2->forward(x);
    x = ln2->forward(x);
    x = torch::relu(x);
    x = dropout->forward(x);
    x = fc3->forward(x);
    return x;
}
