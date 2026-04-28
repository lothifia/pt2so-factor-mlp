#include "state_dict.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

std::string sizes_to_string(c10::IntArrayRef sizes) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << sizes[i];
    }
    oss << "]";
    return oss.str();
}

std::string dtype_to_string(torch::ScalarType dtype) {
    return std::string(c10::toString(dtype));
}

void copy_named_tensor(
    const std::string& name,
    torch::Tensor destination,
    const TensorMap& weights,
    std::unordered_set<std::string>& consumed,
    std::vector<std::string>& errors) {
    auto it = weights.find(name);
    if (it == weights.end()) {
        errors.push_back("missing tensor: " + name);
        return;
    }

    const torch::Tensor& source = it->second;
    if (!source.defined()) {
        errors.push_back("weight tensor is undefined: " + name);
        return;
    }
    if (source.sizes().vec() != destination.sizes().vec()) {
        errors.push_back(
            "shape mismatch for " + name + ": expected " + sizes_to_string(destination.sizes()) +
            ", got " + sizes_to_string(source.sizes()));
        return;
    }
    if (source.scalar_type() != destination.scalar_type()) {
        errors.push_back(
            "dtype mismatch for " + name + ": expected " + dtype_to_string(destination.scalar_type()) +
            ", got " + dtype_to_string(source.scalar_type()));
        return;
    }

    destination.copy_(source);
    consumed.insert(name);
}

}  // namespace

void load_state_dict_strict(torch::nn::Module& model, const TensorMap& weights) {
    std::vector<std::string> errors;
    std::unordered_set<std::string> consumed;
    consumed.reserve(weights.size());

    torch::NoGradGuard no_grad;

    for (const auto& item : model.named_parameters(/*recurse=*/true)) {
        copy_named_tensor(item.key(), item.value(), weights, consumed, errors);
    }

    for (const auto& item : model.named_buffers(/*recurse=*/true)) {
        copy_named_tensor(item.key(), item.value(), weights, consumed, errors);
    }

    for (const auto& item : weights) {
        if (consumed.find(item.first) == consumed.end()) {
            errors.push_back("unexpected tensor: " + item.first);
        }
    }

    if (!errors.empty()) {
        std::ostringstream oss;
        oss << "strict state_dict load failed:";
        for (const std::string& error : errors) {
            oss << "\n  - " << error;
        }
        throw std::runtime_error(oss.str());
    }
}
