#pragma once

#include <cstddef>
#include <cstdint>

const uint8_t* embedded_encrypted_weights_pack_data();
size_t embedded_encrypted_weights_pack_size();

const uint8_t* embedded_weights_key_data();
size_t embedded_weights_key_size();
