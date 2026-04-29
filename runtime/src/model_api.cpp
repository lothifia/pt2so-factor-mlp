#include "model_api.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <openssl/crypto.h>
#include <openssl/evp.h>

#include "blob.h"
#include "factor_mlp.h"
#include "state_dict.h"
#include "weight_pack.h"

namespace {

constexpr uint8_t kEncryptedPackMagic[8] = {'P', 'T', '2', 'S', 'O', '_', 'E', '1'};
constexpr int64_t kNumFeatures = 128;
constexpr int64_t kHidden1 = 256;
constexpr int64_t kHidden2 = 64;
constexpr int64_t kOutputDim = 1;
constexpr double kDropout = 0.1;

thread_local std::string g_last_error;

struct ModelContext {
    FactorMLP model{nullptr};
    std::string last_error;
};

ModelContext* as_context(ModelHandle handle) {
    return reinterpret_cast<ModelContext*>(handle);
}

void set_global_error(const std::string& message) {
    g_last_error = message;
}

int fail(ModelContext* ctx, int code, const std::string& message) {
    if (ctx != nullptr) {
        ctx->last_error = message;
    } else {
        set_global_error(message);
    }
    return code;
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

std::vector<uint8_t> parse_hex_key(const std::string& value, const std::string& source) {
    std::string compact;
    compact.reserve(value.size());
    for (unsigned char c : value) {
        if (!std::isspace(c)) {
            compact.push_back(static_cast<char>(c));
        }
    }

    if (compact.rfind("0x", 0) == 0 || compact.rfind("0X", 0) == 0) {
        compact.erase(0, 2);
    }

    if (compact.size() != 64) {
        throw std::runtime_error(source + " must contain a 64-character AES-256 key in hex");
    }

    std::vector<uint8_t> key(32);
    for (size_t i = 0; i < key.size(); ++i) {
        const int hi = hex_value(compact[2 * i]);
        const int lo = hex_value(compact[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            throw std::runtime_error(source + " contains a non-hex character");
        }
        key[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return key;
}

std::vector<uint8_t> read_all_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("failed to open PT2SO_WEIGHTS_KEY_FILE: " + path);
    }

    const std::streamsize file_size = f.tellg();
    if (file_size < 0) {
        throw std::runtime_error("failed to determine key file size: " + path);
    }
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
    if (!bytes.empty() && !f.read(reinterpret_cast<char*>(bytes.data()), file_size)) {
        throw std::runtime_error("failed to read key file: " + path);
    }
    return bytes;
}

std::vector<uint8_t> load_runtime_key() {
    const char* hex_env = std::getenv("PT2SO_WEIGHTS_KEY_HEX");
    if (hex_env != nullptr && hex_env[0] != '\0') {
        return parse_hex_key(hex_env, "PT2SO_WEIGHTS_KEY_HEX");
    }

    const char* file_env = std::getenv("PT2SO_WEIGHTS_KEY_FILE");
    if (file_env == nullptr || file_env[0] == '\0') {
        throw std::runtime_error(
            "missing AES key: set PT2SO_WEIGHTS_KEY_HEX or PT2SO_WEIGHTS_KEY_FILE before model_create");
    }

    std::vector<uint8_t> key_file = read_all_bytes(file_env);
    if (key_file.size() == 32) {
        return key_file;
    }

    const std::string maybe_hex(key_file.begin(), key_file.end());
    return parse_hex_key(maybe_hex, "PT2SO_WEIGHTS_KEY_FILE");
}

size_t checked_bytes_for_float_tensor(int64_t rows, int64_t cols, const std::string& name) {
    if (rows < 0 || cols < 0) {
        throw std::runtime_error(name + " has negative shape");
    }
    const auto urows = static_cast<size_t>(rows);
    const auto ucols = static_cast<size_t>(cols);
    if (ucols != 0 && urows > std::numeric_limits<size_t>::max() / ucols) {
        throw std::runtime_error(name + " element count overflows size_t");
    }
    const size_t elements = urows * ucols;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::runtime_error(name + " byte count overflows size_t");
    }
    return elements * sizeof(float);
}

void fill_output_desc_shape(TensorDesc& output, int64_t batch) {
    output.dtype = MODEL_FLOAT32;
    output.ndim = 2;
    output.shape[0] = batch;
    output.shape[1] = 1;
    for (int i = 2; i < 8; ++i) {
        output.shape[i] = 0;
    }
    output.bytes = checked_bytes_for_float_tensor(batch, 1, "output");
}

struct EvpCipherCtxDeleter {
    void operator()(EVP_CIPHER_CTX* ctx) const {
        EVP_CIPHER_CTX_free(ctx);
    }
};

using EvpCipherCtxPtr = std::unique_ptr<EVP_CIPHER_CTX, EvpCipherCtxDeleter>;

class ByteReader {
public:
    ByteReader(const uint8_t* data, size_t size) : data_(data), size_(size) {
        if (data == nullptr && size != 0) {
            throw std::runtime_error("embedded encrypted weights pack pointer is null");
        }
    }

    const uint8_t* read_bytes(size_t n, const std::string& what) {
        if (offset_ > size_ || n > size_ - offset_) {
            std::ostringstream oss;
            oss << "truncated embedded encrypted weights pack while reading " << what;
            throw std::runtime_error(oss.str());
        }
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

    size_t remaining() const {
        return size_ - offset_;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t offset_{0};
};

struct EncryptedWeightsView {
    const uint8_t* nonce{nullptr};
    size_t nonce_size{0};
    const uint8_t* ciphertext_and_tag{nullptr};
    size_t ciphertext_and_tag_size{0};
};

EncryptedWeightsView parse_embedded_encrypted_pack() {
    const uint8_t* pack = embedded_encrypted_weights_pack_data();
    const size_t pack_size = embedded_encrypted_weights_pack_size();
    if (pack == nullptr || pack_size == 0) {
        throw std::runtime_error("embedded encrypted weights pack is empty");
    }

    ByteReader reader(pack, pack_size);
    const uint8_t* magic = reader.read_bytes(8, "magic");
    if (std::memcmp(magic, kEncryptedPackMagic, sizeof(kEncryptedPackMagic)) != 0) {
        throw std::runtime_error("invalid embedded encrypted weights pack magic");
    }

    const uint32_t nonce_size = reader.read_u32("nonce_len");
    const uint8_t* nonce = reader.read_bytes(nonce_size, "nonce");
    const uint64_t ciphertext_size_u64 = reader.read_u64("ciphertext_len");
    if (ciphertext_size_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("embedded ciphertext length overflows size_t");
    }
    const size_t ciphertext_size = static_cast<size_t>(ciphertext_size_u64);
    const uint8_t* ciphertext_and_tag = reader.read_bytes(ciphertext_size, "ciphertext_and_tag");
    if (reader.remaining() != 0) {
        throw std::runtime_error("embedded encrypted weights pack has trailing bytes");
    }
    if (nonce_size != 12) {
        throw std::runtime_error("embedded AES-GCM nonce must be exactly 12 bytes");
    }
    if (ciphertext_size <= 16) {
        throw std::runtime_error("embedded ciphertext must include a 16-byte GCM tag");
    }

    return EncryptedWeightsView{
        nonce,
        nonce_size,
        ciphertext_and_tag,
        ciphertext_size,
    };
}

std::vector<uint8_t> decrypt_embedded_weights() {
    const EncryptedWeightsView encrypted_pack = parse_embedded_encrypted_pack();
    const uint8_t* encrypted = encrypted_pack.ciphertext_and_tag;
    const size_t encrypted_size = encrypted_pack.ciphertext_and_tag_size;
    std::vector<uint8_t> key = load_runtime_key();
    const uint8_t* nonce = encrypted_pack.nonce;
    const size_t nonce_size = encrypted_pack.nonce_size;

    if (key.size() != 32) {
        throw std::runtime_error("runtime AES-256-GCM key must be exactly 32 bytes");
    }
    if (encrypted_size - 16 > static_cast<size_t>(std::numeric_limits<int>::max())) {
        OPENSSL_cleanse(key.data(), key.size());
        throw std::runtime_error("embedded encrypted weights blob is too large for OpenSSL EVP");
    }

    constexpr unsigned char kAad[] = "PT2SO_FACTOR_MLP_V1";
    const size_t ciphertext_size = encrypted_size - 16;
    const uint8_t* tag = encrypted + ciphertext_size;

    EvpCipherCtxPtr ctx(EVP_CIPHER_CTX_new());
    if (!ctx) {
        OPENSSL_cleanse(key.data(), key.size());
        throw std::runtime_error("failed to allocate OpenSSL EVP_CIPHER_CTX");
    }

    if (EVP_DecryptInit_ex(ctx.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1) {
        OPENSSL_cleanse(key.data(), key.size());
        throw std::runtime_error("OpenSSL EVP_DecryptInit_ex failed");
    }
    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, static_cast<int>(nonce_size), nullptr) != 1) {
        OPENSSL_cleanse(key.data(), key.size());
        throw std::runtime_error("OpenSSL failed to set AES-GCM nonce length");
    }
    if (EVP_DecryptInit_ex(ctx.get(), nullptr, nullptr, key.data(), nonce) != 1) {
        OPENSSL_cleanse(key.data(), key.size());
        throw std::runtime_error("OpenSSL failed to initialize AES-GCM key and nonce");
    }

    OPENSSL_cleanse(key.data(), key.size());

    int out_len = 0;
    if (EVP_DecryptUpdate(
            ctx.get(),
            nullptr,
            &out_len,
            kAad,
            static_cast<int>(sizeof(kAad) - 1)) != 1) {
        throw std::runtime_error("OpenSSL failed to authenticate AES-GCM AAD");
    }

    std::vector<uint8_t> plaintext(ciphertext_size);
    int plaintext_len = 0;
    if (EVP_DecryptUpdate(
            ctx.get(),
            plaintext.data(),
            &out_len,
            encrypted,
            static_cast<int>(ciphertext_size)) != 1) {
        OPENSSL_cleanse(plaintext.data(), plaintext.size());
        throw std::runtime_error("OpenSSL AES-GCM decrypt update failed");
    }
    plaintext_len += out_len;

    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_TAG, 16, const_cast<uint8_t*>(tag)) != 1) {
        OPENSSL_cleanse(plaintext.data(), plaintext.size());
        throw std::runtime_error("OpenSSL failed to set AES-GCM tag");
    }

    if (EVP_DecryptFinal_ex(ctx.get(), plaintext.data() + plaintext_len, &out_len) != 1) {
        OPENSSL_cleanse(plaintext.data(), plaintext.size());
        throw std::runtime_error("embedded weights authentication failed");
    }
    plaintext_len += out_len;
    plaintext.resize(static_cast<size_t>(plaintext_len));
    return plaintext;
}

}  // namespace

extern "C" {

int model_create(ModelHandle* out_handle) {
    if (out_handle == nullptr) {
        return fail(nullptr, -1, "model_create received null out_handle");
    }
    *out_handle = nullptr;

    try {
        auto ctx = std::make_unique<ModelContext>();
        ctx->model = FactorMLP(kNumFeatures, kHidden1, kHidden2, kOutputDim, kDropout);

        std::vector<uint8_t> decrypted_weights = decrypt_embedded_weights();
        TensorMap weights;
        try {
            weights = parse_weight_pack(decrypted_weights.data(), decrypted_weights.size());
        } catch (...) {
            OPENSSL_cleanse(decrypted_weights.data(), decrypted_weights.size());
            throw;
        }
        OPENSSL_cleanse(decrypted_weights.data(), decrypted_weights.size());
        load_state_dict_strict(*ctx->model, weights);
        ctx->model->eval();
        ctx->last_error.clear();

        *out_handle = reinterpret_cast<ModelHandle>(ctx.release());
        g_last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        return fail(nullptr, -2, e.what());
    } catch (...) {
        return fail(nullptr, -2, "unknown exception in model_create");
    }
}

int model_forward(
    ModelHandle handle,
    const TensorDesc* inputs,
    int input_count,
    TensorDesc* outputs,
    int output_count) {
    ModelContext* ctx = as_context(handle);
    if (ctx == nullptr) {
        return fail(nullptr, -1, "model_forward received null handle");
    }

    try {
        if (inputs == nullptr) {
            return fail(ctx, -1, "model_forward received null inputs");
        }
        if (outputs == nullptr) {
            return fail(ctx, -1, "model_forward received null outputs");
        }
        if (input_count != 1) {
            return fail(ctx, -1, "model_forward expects exactly 1 input");
        }
        if (output_count != 1) {
            return fail(ctx, -1, "model_forward expects exactly 1 output");
        }

        const TensorDesc& input = inputs[0];
        TensorDesc& output = outputs[0];

        if (input.dtype != MODEL_FLOAT32) {
            return fail(ctx, -1, "model_forward input dtype must be MODEL_FLOAT32");
        }
        if (input.ndim != 2) {
            return fail(ctx, -1, "model_forward input rank must be 2");
        }
        const int64_t batch = input.shape[0];
        const int64_t features = input.shape[1];
        if (batch < 0) {
            return fail(ctx, -1, "model_forward input batch dimension must be non-negative");
        }
        if (features != kNumFeatures) {
            std::ostringstream oss;
            oss << "model_forward input feature dimension must be " << kNumFeatures
                << ", got " << features;
            return fail(ctx, -1, oss.str());
        }

        const size_t expected_input_bytes = checked_bytes_for_float_tensor(batch, features, "input");
        if (input.bytes < expected_input_bytes) {
            std::ostringstream oss;
            oss << "model_forward input buffer is too small: need " << expected_input_bytes
                << " bytes, got " << input.bytes;
            return fail(ctx, -1, oss.str());
        }
        if (expected_input_bytes > 0 && input.data == nullptr) {
            return fail(ctx, -1, "model_forward input data is null");
        }

        torch::InferenceMode guard;
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor x;
        if (expected_input_bytes == 0) {
            x = torch::empty({batch, kNumFeatures}, options);
        } else {
            x = torch::from_blob(
                input.data,
                {batch, kNumFeatures},
                options);
        }

        torch::Tensor y = ctx->model->forward(x);
        y = y.to(torch::kCPU).to(torch::kFloat32).contiguous();
        if (y.dim() != 2 || y.size(0) != batch || y.size(1) != 1) {
            throw std::runtime_error("model produced an unexpected output shape");
        }

        void* output_data = output.data;
        const size_t output_capacity = output.bytes;
        fill_output_desc_shape(output, batch);
        const size_t required_output_bytes = output.bytes;
        if ((required_output_bytes > 0 && output_data == nullptr) || output_capacity < required_output_bytes) {
            std::ostringstream oss;
            oss << "model_forward output buffer is null or too small: need "
                << required_output_bytes << " bytes, got " << output_capacity;
            return fail(ctx, -3, oss.str());
        }

        std::memcpy(output_data, y.data_ptr(), required_output_bytes);
        output.bytes = required_output_bytes;
        ctx->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        return fail(ctx, -2, e.what());
    } catch (...) {
        return fail(ctx, -2, "unknown exception in model_forward");
    }
}

void model_destroy(ModelHandle handle) {
    delete as_context(handle);
}

const char* model_last_error(ModelHandle handle) {
    ModelContext* ctx = as_context(handle);
    if (ctx == nullptr) {
        return g_last_error.c_str();
    }
    return ctx->last_error.c_str();
}

const char* model_version() {
    return "pt2so-factor-mlp/0.1.0";
}

}
