#pragma once

#include <torch/torch.h>

#include "weight_pack.h"

void load_state_dict_strict(torch::nn::Module& model, const TensorMap& weights);
