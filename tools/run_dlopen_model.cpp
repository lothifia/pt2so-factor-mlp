#include "../runtime/include/model_api.h"

#include <dlfcn.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct FloatArray {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

std::vector<uint8_t> read_all_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("failed to open file: " + path);
    }
    const std::streamsize file_size = f.tellg();
    if (file_size < 0) {
        throw std::runtime_error("failed to determine file size: " + path);
    }
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
    if (!bytes.empty() && !f.read(reinterpret_cast<char*>(bytes.data()), file_size)) {
        throw std::runtime_error("failed to read file: " + path);
    }
    return bytes;
}

uint16_t read_le16(const std::vector<uint8_t>& bytes, size_t offset) {
    if (offset + 2 > bytes.size()) {
        throw std::runtime_error("truncated npy header length");
    }
    return static_cast<uint16_t>(bytes[offset]) |
           static_cast<uint16_t>(static_cast<uint16_t>(bytes[offset + 1]) << 8);
}

uint32_t read_le32(const std::vector<uint8_t>& bytes, size_t offset) {
    if (offset + 4 > bytes.size()) {
        throw std::runtime_error("truncated npy header length");
    }
    return static_cast<uint32_t>(bytes[offset]) |
           (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

std::string trim(const std::string& value) {
    const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (begin >= end) {
        return "";
    }
    return std::string(begin, end);
}

std::vector<int64_t> parse_shape(const std::string& header) {
    const size_t shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) {
        throw std::runtime_error("npy header is missing shape");
    }
    const size_t open = header.find('(', shape_pos);
    const size_t close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos || close <= open) {
        throw std::runtime_error("npy header has invalid shape tuple");
    }

    std::vector<int64_t> shape;
    std::stringstream ss(header.substr(open + 1, close - open - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (!token.empty()) {
            shape.push_back(std::stoll(token));
        }
    }
    if (shape.empty()) {
        throw std::runtime_error("npy shape must not be scalar");
    }
    return shape;
}

size_t checked_element_count(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("npy shape contains a negative dimension");
        }
        const size_t udim = static_cast<size_t>(dim);
        if (udim != 0 && count > std::numeric_limits<size_t>::max() / udim) {
            throw std::runtime_error("npy element count overflow");
        }
        count *= udim;
    }
    return count;
}

FloatArray load_npy_float32(const std::string& path) {
    const std::vector<uint8_t> bytes = read_all_bytes(path);
    constexpr uint8_t kMagic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (bytes.size() < 10 || std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("not a numpy .npy file: " + path);
    }

    const uint8_t major = bytes[6];
    size_t header_len = 0;
    size_t payload_offset = 0;
    if (major == 1) {
        header_len = read_le16(bytes, 8);
        payload_offset = 10 + header_len;
    } else if (major == 2 || major == 3) {
        header_len = read_le32(bytes, 8);
        payload_offset = 12 + header_len;
    } else {
        throw std::runtime_error("unsupported npy version");
    }
    if (payload_offset > bytes.size()) {
        throw std::runtime_error("truncated npy payload");
    }

    const size_t header_offset = major == 1 ? 10 : 12;
    const std::string header(
        reinterpret_cast<const char*>(bytes.data() + header_offset),
        header_len);

    if (header.find("'descr'") == std::string::npos ||
        (header.find("'<f4'") == std::string::npos && header.find("'|f4'") == std::string::npos)) {
        throw std::runtime_error("npy input must be float32 little-endian");
    }
    if (header.find("'fortran_order'") == std::string::npos ||
        header.find("False") == std::string::npos) {
        throw std::runtime_error("npy input must be C-contiguous, not Fortran-order");
    }

    FloatArray array;
    array.shape = parse_shape(header);
    const size_t element_count = checked_element_count(array.shape);
    if (element_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::runtime_error("npy byte count overflow");
    }
    const size_t data_bytes = element_count * sizeof(float);
    if (data_bytes > bytes.size() - payload_offset) {
        throw std::runtime_error("truncated npy float32 data");
    }
    array.data.resize(element_count);
    if (data_bytes != 0) {
        std::memcpy(array.data.data(), bytes.data() + payload_offset, data_bytes);
    }
    return array;
}

std::string shape_tuple_for_header(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    if (shape.size() == 1) {
        oss << ",";
    }
    oss << ")";
    return oss.str();
}

void save_npy_float32(
    const std::string& path,
    const std::vector<int64_t>& shape,
    const std::vector<float>& data) {
    const size_t element_count = checked_element_count(shape);
    if (element_count != data.size()) {
        throw std::runtime_error("output shape does not match output data size");
    }

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " +
                         shape_tuple_for_header(shape) + ", }";
    constexpr size_t kPreambleSize = 10;
    while ((kPreambleSize + header.size() + 1) % 16 != 0) {
        header.push_back(' ');
    }
    header.push_back('\n');
    if (header.size() > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("npy header too large for version 1.0");
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    const uint8_t version[] = {1, 0};
    const auto header_len = static_cast<uint16_t>(header.size());
    const uint8_t header_len_bytes[] = {
        static_cast<uint8_t>(header_len & 0xff),
        static_cast<uint8_t>((header_len >> 8) & 0xff),
    };

    f.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    f.write(reinterpret_cast<const char*>(version), sizeof(version));
    f.write(reinterpret_cast<const char*>(header_len_bytes), sizeof(header_len_bytes));
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!data.empty()) {
        f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    }
    if (!f) {
        throw std::runtime_error("failed to write output file: " + path);
    }
}

template <typename T>
T load_symbol(void* library, const char* name) {
    dlerror();
    void* symbol = dlsym(library, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        throw std::runtime_error(std::string("failed to load symbol ") + name + ": " + (error ? error : "null"));
    }
    return reinterpret_cast<T>(symbol);
}

void fill_tensor_desc(TensorDesc& desc, FloatArray& array) {
    if (array.shape.size() > 8) {
        throw std::runtime_error("TensorDesc supports at most 8 dimensions");
    }
    desc.data = array.data.empty() ? nullptr : array.data.data();
    desc.dtype = MODEL_FLOAT32;
    desc.ndim = static_cast<int>(array.shape.size());
    for (int i = 0; i < 8; ++i) {
        desc.shape[i] = i < desc.ndim ? array.shape[static_cast<size_t>(i)] : 0;
    }
    desc.bytes = array.data.size() * sizeof(float);
}

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [--lib build/libmodel.so]"
        << " [--input artifacts/factor_mlp/sample_input.npy]"
        << " [--output artifacts/factor_mlp/cpp_dlopen_output.npy]\n\n"
        << "Compile example:\n"
        << "  g++ -std=c++17 -O2 -Iruntime/include tools/run_dlopen_model.cpp -ldl -o build/run_dlopen_model\n\n"
        << "Runtime key example:\n"
        << "  export PT2SO_MODEL_KEY_FILE=artifacts/factor_mlp/model.key\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string lib_path = "build/libmodel.so";
    std::string input_path = "artifacts/factor_mlp/sample_input.npy";
    std::string output_path = "artifacts/factor_mlp/cpp_dlopen_output.npy";
    void* library = nullptr;
    ModelHandle model = nullptr;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const char* name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(std::string("missing value for ") + name);
                }
                return argv[++i];
            };

            if (arg == "--lib") {
                lib_path = require_value("--lib");
            } else if (arg == "--input") {
                input_path = require_value("--input");
            } else if (arg == "--output") {
                output_path = require_value("--output");
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        FloatArray input = load_npy_float32(input_path);
        if (input.shape.size() != 2 || input.shape[1] != 128) {
            throw std::runtime_error("input sample must have shape [B, 128]");
        }

        library = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) {
            const char* error = dlerror();
            throw std::runtime_error(std::string("dlopen failed: ") + (error ? error : "unknown error"));
        }

        using ModelCreateFn = int (*)(ModelHandle*);
        using ModelForwardFn = int (*)(ModelHandle, const TensorDesc*, int, TensorDesc*, int);
        using ModelDestroyFn = void (*)(ModelHandle);
        using ModelLastErrorFn = const char* (*)(ModelHandle);
        using ModelVersionFn = const char* (*)();

        const auto model_create = load_symbol<ModelCreateFn>(library, "model_create");
        const auto model_forward = load_symbol<ModelForwardFn>(library, "model_forward");
        const auto model_destroy = load_symbol<ModelDestroyFn>(library, "model_destroy");
        const auto model_last_error = load_symbol<ModelLastErrorFn>(library, "model_last_error");
        const auto model_version = load_symbol<ModelVersionFn>(library, "model_version");

        std::cout << "model_version=" << model_version() << "\n";

        int rc = model_create(&model);
        if (rc != 0) {
            const char* error = model_last_error(nullptr);
            throw std::runtime_error("model_create failed rc=" + std::to_string(rc) + ": " + (error ? error : ""));
        }

        TensorDesc input_desc{};
        fill_tensor_desc(input_desc, input);

        TensorDesc output_probe{};
        rc = model_forward(model, &input_desc, 1, &output_probe, 1);
        if (rc != -3) {
            const char* error = model_last_error(model);
            throw std::runtime_error("expected output-size probe rc=-3, got rc=" + std::to_string(rc) +
                                     ": " + (error ? error : ""));
        }

        if (output_probe.ndim != 2 || output_probe.shape[0] != input.shape[0] || output_probe.shape[1] != 1) {
            throw std::runtime_error("model reported an unexpected output shape");
        }
        if (output_probe.bytes % sizeof(float) != 0) {
            throw std::runtime_error("model reported output bytes not divisible by float size");
        }

        std::vector<float> output(output_probe.bytes / sizeof(float));
        TensorDesc output_desc{};
        output_desc.data = output.empty() ? nullptr : output.data();
        output_desc.dtype = MODEL_FLOAT32;
        output_desc.bytes = output.size() * sizeof(float);

        rc = model_forward(model, &input_desc, 1, &output_desc, 1);
        if (rc != 0) {
            const char* error = model_last_error(model);
            throw std::runtime_error("model_forward failed rc=" + std::to_string(rc) + ": " + (error ? error : ""));
        }

        std::vector<int64_t> output_shape;
        for (int i = 0; i < output_desc.ndim; ++i) {
            output_shape.push_back(output_desc.shape[i]);
        }
        save_npy_float32(output_path, output_shape, output);
        std::cout << "cpp_output=" << output_path << "\n";
        std::cout << "shape=(" << output_shape[0] << ", " << output_shape[1] << ")\n";

        model_destroy(model);
        model = nullptr;
        dlclose(library);
        library = nullptr;
        return 0;
    } catch (const std::exception& e) {
        if (model != nullptr && library != nullptr) {
            try {
                const auto model_destroy = load_symbol<void (*)(ModelHandle)>(library, "model_destroy");
                model_destroy(model);
            } catch (...) {
            }
        }
        if (library != nullptr) {
            dlclose(library);
        }
        std::cerr << "error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
}
