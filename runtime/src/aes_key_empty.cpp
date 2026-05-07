#include "embedded_key.h"

#include <stdexcept>

void pt2so_materialize_aes_128_key(uint8_t*, size_t) {
    throw std::runtime_error("embedded AES-128 key source has not been generated");
}
