#include "weight_pack.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr uint8_t kMagic[8] = {'P', 'T', '2', 'S', 'O', '_', 'W', '1'};

class Reader {
public:
    Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {
        if (data == nullptr && size != 0) {
            throw std::runtime_error("weight pack data pointer is null");
        }
    }

    void require(size_t n, const std::string& what) const {
        if (offset_ > size_ || n > size_ - offset_) {
            std::ostringstream oss;
            oss << "truncated weight pack while reading " << what
                << " at offset " << offset_ << " need " << n
                << " bytes, have " << (offset_ <= size_ ? size_ - offset_ : 0);
            throw std::runtime_error(oss.str());
        }
    }

    const uint8_t* read_bytes(size_t n, const std::string& what) {
        require(n, what);
        const uint8_t* out = data_ + offset_;
        offset_ += n;
        return out;
    }

    uint32_t read_u32(const std::string& what) {
        const uint8_t* p = read_bytes(4, what);
        return static_cast<uint32_t>(p[0]) |
               (static_cast<uint32_t>(p[1]) << 8) |
               (static_cast<uint32_t>(p[2]) << 16) |
               (static_cast<uint32_t>(p[3]) << 24);
    }

    uint64_t read_u64(const std::string& what) {
        const uint8_t* p = read_bytes(8, what);
        uint64_t value = 0;
        for (int i = 0; i < 8; ++i) {
            value |= static_cast<uint64_t>(p[i]) << (8 * i);
        }
        return value;
    }

    int64_t read_i64(const std::string& what) {
        const uint64_t raw = read_u64(what);
        if (raw <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return static_cast<int64_t>(raw);
        }
        return static_cast<int64_t>(raw - (uint64_t{1} << 63)) + std::numeric_limits<int64_t>::min();
    }

    size_t offset() const { return offset_; }
    size_t remaining() const { return size_ - offset_; }

private:
    const uint8_t* data_;
    size_t size_;
    size_t offset_{0};
};

torch::ScalarType dtype_from_id(uint32_t dtype_id) {
    switch (dtype_id) {
        case 1:
            return torch::kFloat32;
        case 2:
            return torch::kFloat16;
        case 3:
            return torch::kFloat64;
        case 4:
            return torch::kInt64;
        case 5:
            return torch::kInt32;
        case 6:
            return torch::kUInt8;
        case 7:
            return torch::kBool;
        default:
            throw std::runtime_error("unsupported dtype_id " + std::to_string(dtype_id));
    }
}

size_t element_size(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32:
            return 4;
        case torch::kFloat16:
            return 2;
        case torch::kFloat64:
            return 8;
        case torch::kInt64:
            return 8;
        case torch::kInt32:
            return 4;
        case torch::kUInt8:
            return 1;
        case torch::kBool:
            return 1;
        default:
            throw std::runtime_error("unsupported tensor scalar type");
    }
}

size_t checked_numel(const std::vector<int64_t>& shape, const std::string& name) {
    size_t numel = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("tensor " + name + " has negative shape dimension");
        }
        const auto udim = static_cast<size_t>(dim);
        if (udim != 0 && numel > std::numeric_limits<size_t>::max() / udim) {
            throw std::runtime_error("tensor " + name + " shape element count overflows size_t");
        }
        numel *= udim;
    }
    return numel;
}

size_t checked_nbytes(const std::vector<int64_t>& shape, torch::ScalarType dtype, const std::string& name) {
    const size_t numel = checked_numel(shape, name);
    const size_t item_size = element_size(dtype);
    if (item_size != 0 && numel > std::numeric_limits<size_t>::max() / item_size) {
        throw std::runtime_error("tensor " + name + " byte count overflows size_t");
    }
    return numel * item_size;
}

}  // namespace

TensorMap parse_weight_pack(const uint8_t* data, size_t size) {
    Reader reader(data, size);

    const uint8_t* magic = reader.read_bytes(8, "magic");
    if (!std::equal(kMagic, kMagic + 8, magic)) {
        throw std::runtime_error("invalid weight pack magic");
    }

    const uint32_t tensor_count = reader.read_u32("tensor_count");
    TensorMap tensors;
    tensors.reserve(tensor_count);

    for (uint32_t i = 0; i < tensor_count; ++i) {
        const uint32_t name_len = reader.read_u32("name_len");
        const uint8_t* name_bytes = reader.read_bytes(name_len, "tensor name");
        const std::string name(reinterpret_cast<const char*>(name_bytes), name_len);
        if (name.empty()) {
            throw std::runtime_error("encountered tensor with empty name");
        }

        const uint32_t dtype_id = reader.read_u32("dtype_id for " + name);
        const torch::ScalarType dtype = dtype_from_id(dtype_id);

        const uint32_t ndim = reader.read_u32("ndim for " + name);
        if (ndim > 8) {
            throw std::runtime_error("tensor " + name + " has ndim > 8");
        }

        std::vector<int64_t> shape;
        shape.reserve(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            shape.push_back(reader.read_i64("shape for " + name));
        }

        const uint64_t raw_nbytes_u64 = reader.read_u64("raw_nbytes for " + name);
        if (raw_nbytes_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error("tensor " + name + " raw_nbytes overflows size_t");
        }
        const size_t raw_nbytes = static_cast<size_t>(raw_nbytes_u64);
        const size_t expected_nbytes = checked_nbytes(shape, dtype, name);
        if (raw_nbytes != expected_nbytes) {
            std::ostringstream oss;
            oss << "tensor " << name << " raw byte count mismatch: expected "
                << expected_nbytes << ", got " << raw_nbytes;
            throw std::runtime_error(oss.str());
        }

        const uint8_t* raw = reader.read_bytes(raw_nbytes, "raw tensor data for " + name);
        auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
        torch::Tensor tensor = torch::from_blob(
                                   const_cast<uint8_t*>(raw),
                                   torch::IntArrayRef(shape.data(), shape.size()),
                                   options)
                                   .clone();

        auto inserted = tensors.emplace(name, tensor);
        if (!inserted.second) {
            throw std::runtime_error("duplicate tensor name in weight pack: " + name);
        }
    }

    if (reader.remaining() != 0) {
        std::ostringstream oss;
        oss << "weight pack has " << reader.remaining()
            << " trailing bytes after " << tensor_count << " tensors";
        throw std::runtime_error(oss.str());
    }

    return tensors;
}
