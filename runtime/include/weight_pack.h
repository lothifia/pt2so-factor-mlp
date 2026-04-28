#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include <torch/torch.h>

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

TensorMap parse_weight_pack(const uint8_t* data, size_t size);
