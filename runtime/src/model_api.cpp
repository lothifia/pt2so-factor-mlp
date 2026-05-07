#include "model_api.h"

#include <torch/script.h>

#include <openssl/crypto.h>
#include <openssl/evp.h>

#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blob.h"
#include "embedded_key.h"

namespace {

constexpr uint8_t kEncryptedModelMagicV2[8] = {'P', 'T', '2', 'S', 'O', '_', 'E', '2'};
constexpr uint8_t kPlaintextModelMagicV1[8] = {'P', 'T', '2', 'S', 'O', '_', 'P', '1'};
constexpr uint32_t kAes128GcmAlgorithmId = 7;
constexpr int64_t kNumFeatures = 128;
constexpr size_t kAes128KeyBytes = 16;
constexpr size_t kGcmNonceBytes = 12;
constexpr size_t kGcmTagBytes = 16;
constexpr unsigned char kAad[] = "PT2SO_TORCHSCRIPT_V1";

using Clock = std::chrono::steady_clock;

thread_local std::string g_last_error;

struct CreateTiming {
    std::string algorithm;
    size_t encrypted_bytes{0};
    size_t plaintext_bytes{0};
    double decrypt_ms{0.0};
    double jit_load_ms{0.0};
    double total_ms{0.0};
};

struct ModelContext {
    std::unique_ptr<torch::jit::Module> model;
    std::string last_error;
    CreateTiming create_timing;
};

struct ModelBlob {
    std::vector<uint8_t> plaintext;
    std::string algorithm;
    size_t encrypted_pack_bytes{0};
    size_t plaintext_bytes{0};
    bool encrypted{true};
};

struct EncryptedModelView {
    const uint8_t* nonce{nullptr};
    size_t nonce_size{0};
    const uint8_t* ciphertext_and_tag{nullptr};
    size_t ciphertext_and_tag_size{0};
    size_t pack_size{0};
};

struct EvpCipherCtxDeleter {
    void operator()(EVP_CIPHER_CTX* ctx) const {
        EVP_CIPHER_CTX_free(ctx);
    }
};

using EvpCipherCtxPtr = std::unique_ptr<EVP_CIPHER_CTX, EvpCipherCtxDeleter>;

ModelContext* as_context(ModelHandle handle) {
    return reinterpret_cast<ModelContext*>(handle);
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
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

void init_openssl_crypto() {
    static std::once_flag once;
    std::call_once(once, []() {
        if (OPENSSL_init_crypto(OPENSSL_INIT_NO_LOAD_CONFIG, nullptr) != 1) {
            throw std::runtime_error("OpenSSL crypto initialization failed");
        }
    });
}

void secure_clear(void* data, size_t size) {
    if (data == nullptr || size == 0) {
        return;
    }
    OPENSSL_cleanse(data, size);
}

void secure_clear(std::vector<uint8_t>& bytes) {
    if (!bytes.empty()) {
        secure_clear(bytes.data(), bytes.size());
    }
}

void secure_clear(std::string& value) {
    if (!value.empty()) {
        secure_clear(&value[0], value.size());
    }
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

class ByteReader {
public:
    ByteReader(const uint8_t* data, size_t size) : data_(data), size_(size) {
        if (data == nullptr && size != 0) {
            throw std::runtime_error("embedded model pointer is null");
        }
    }

    const uint8_t* read_bytes(size_t n, const std::string& what) {
        if (offset_ > size_ || n > size_ - offset_) {
            std::ostringstream oss;
            oss << "truncated embedded model while reading " << what;
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

bool has_magic(const uint8_t* data, size_t size, const uint8_t (&magic)[8]) {
    return size >= sizeof(magic) && std::memcmp(data, magic, sizeof(magic)) == 0;
}

EncryptedModelView parse_embedded_encrypted_model(const uint8_t* pack, size_t pack_size) {
    ByteReader reader(pack, pack_size);
    const uint8_t* magic = reader.read_bytes(8, "magic");
    if (std::memcmp(magic, kEncryptedModelMagicV2, sizeof(kEncryptedModelMagicV2)) != 0) {
        throw std::runtime_error("invalid embedded AES-GCM model magic");
    }

    const uint32_t algorithm_id = reader.read_u32("algorithm_id");
    if (algorithm_id != kAes128GcmAlgorithmId) {
        throw std::runtime_error("unsupported encrypted model algorithm id: " + std::to_string(algorithm_id));
    }

    const uint32_t nonce_size = reader.read_u32("nonce_len");
    if (nonce_size != kGcmNonceBytes) {
        std::ostringstream oss;
        oss << "gcm-aes-128 nonce must be exactly " << kGcmNonceBytes
            << " bytes, got " << nonce_size;
        throw std::runtime_error(oss.str());
    }
    const uint8_t* nonce = reader.read_bytes(nonce_size, "nonce");

    const uint64_t ciphertext_size_u64 = reader.read_u64("ciphertext_len");
    if (ciphertext_size_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("embedded ciphertext length overflows size_t");
    }
    const size_t ciphertext_size = static_cast<size_t>(ciphertext_size_u64);
    if (ciphertext_size < kGcmTagBytes) {
        throw std::runtime_error("AES-128-GCM ciphertext must include a 16-byte authentication tag");
    }
    const uint8_t* ciphertext_and_tag = reader.read_bytes(ciphertext_size, "ciphertext_and_tag");
    if (reader.remaining() != 0) {
        throw std::runtime_error("embedded AES-GCM model has trailing bytes");
    }

    return EncryptedModelView{
        nonce,
        nonce_size,
        ciphertext_and_tag,
        ciphertext_size,
        pack_size,
    };
}

int checked_int_size(size_t value, const std::string& name) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(name + " is too large for OpenSSL EVP");
    }
    return static_cast<int>(value);
}

std::vector<uint8_t> decrypt_aes_128_gcm(const EncryptedModelView& encrypted_model) {
    init_openssl_crypto();

    std::array<uint8_t, kAes128KeyBytes> key{};
    pt2so_materialize_aes_128_key(key.data(), key.size());

    const uint8_t* encrypted = encrypted_model.ciphertext_and_tag;
    const size_t encrypted_size = encrypted_model.ciphertext_and_tag_size;
    const size_t ciphertext_size = encrypted_size - kGcmTagBytes;
    const uint8_t* tag = encrypted + ciphertext_size;

    EvpCipherCtxPtr ctx(EVP_CIPHER_CTX_new());
    if (!ctx) {
        secure_clear(key.data(), key.size());
        throw std::runtime_error("failed to allocate OpenSSL EVP_CIPHER_CTX");
    }

    auto fail_with_key_clear = [&](const std::string& message) {
        secure_clear(key.data(), key.size());
        throw std::runtime_error(message);
    };

    if (EVP_DecryptInit_ex(ctx.get(), EVP_aes_128_gcm(), nullptr, nullptr, nullptr) != 1) {
        fail_with_key_clear("OpenSSL AES-128-GCM decrypt init failed");
    }
    if (EVP_CIPHER_CTX_ctrl(
            ctx.get(),
            EVP_CTRL_GCM_SET_IVLEN,
            checked_int_size(encrypted_model.nonce_size, "nonce"),
            nullptr) != 1) {
        fail_with_key_clear("OpenSSL failed to set AES-128-GCM nonce length");
    }
    if (EVP_DecryptInit_ex(ctx.get(), nullptr, nullptr, key.data(), encrypted_model.nonce) != 1) {
        fail_with_key_clear("OpenSSL failed to initialize AES-128-GCM key and nonce");
    }

    int out_len = 0;
    if (EVP_DecryptUpdate(
            ctx.get(),
            nullptr,
            &out_len,
            kAad,
            static_cast<int>(sizeof(kAad) - 1)) != 1) {
        fail_with_key_clear("OpenSSL failed to authenticate AES-128-GCM AAD");
    }

    std::vector<uint8_t> plaintext(ciphertext_size);
    int plaintext_len = 0;
    if (EVP_DecryptUpdate(
            ctx.get(),
            plaintext.data(),
            &out_len,
            encrypted,
            checked_int_size(ciphertext_size, "ciphertext")) != 1) {
        secure_clear(plaintext);
        fail_with_key_clear("OpenSSL AES-128-GCM decrypt update failed");
    }
    plaintext_len += out_len;

    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_TAG, kGcmTagBytes, const_cast<uint8_t*>(tag)) != 1) {
        secure_clear(plaintext);
        fail_with_key_clear("OpenSSL failed to set AES-128-GCM tag");
    }

    if (EVP_DecryptFinal_ex(ctx.get(), plaintext.data() + plaintext_len, &out_len) != 1) {
        secure_clear(plaintext);
        fail_with_key_clear("embedded model authentication failed");
    }
    plaintext_len += out_len;
    plaintext.resize(static_cast<size_t>(plaintext_len));
    secure_clear(key.data(), key.size());
    return plaintext;
}

ModelBlob load_embedded_model_blob() {
    const uint8_t* embedded_model = embedded_encrypted_model_data();
    const size_t embedded_model_size = embedded_encrypted_model_size();
    if (embedded_model == nullptr || embedded_model_size == 0) {
        throw std::runtime_error("embedded model blob is empty");
    }

    if (has_magic(embedded_model, embedded_model_size, kPlaintextModelMagicV1)) {
        ByteReader reader(embedded_model, embedded_model_size);
        (void)reader.read_bytes(8, "plaintext magic");
        const uint64_t plaintext_size_u64 = reader.read_u64("plaintext_len");
        if (plaintext_size_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error("embedded plaintext length overflows size_t");
        }
        const size_t plaintext_size = static_cast<size_t>(plaintext_size_u64);
        const uint8_t* plaintext = reader.read_bytes(plaintext_size, "plaintext TorchScript model");
        if (reader.remaining() != 0) {
            throw std::runtime_error("embedded plaintext model has trailing bytes");
        }

        ModelBlob result;
        result.algorithm = "plaintext-embedded";
        result.encrypted = false;
        result.encrypted_pack_bytes = 0;
        result.plaintext_bytes = plaintext_size;
        result.plaintext.assign(plaintext, plaintext + plaintext_size);
        return result;
    }

    const EncryptedModelView encrypted_model = parse_embedded_encrypted_model(embedded_model, embedded_model_size);

    ModelBlob result;
    result.algorithm = "gcm-aes-128";
    result.encrypted = true;
    result.encrypted_pack_bytes = encrypted_model.pack_size;
    result.plaintext = decrypt_aes_128_gcm(encrypted_model);
    result.plaintext_bytes = result.plaintext.size();
    return result;
}

std::unique_ptr<torch::jit::Module> load_jit_model_from_memory(const std::vector<uint8_t>& model_bytes) {
    std::string model_string(reinterpret_cast<const char*>(model_bytes.data()), model_bytes.size());
    try {
        std::istringstream model_stream(model_string, std::ios::in | std::ios::binary);
        auto module = std::make_unique<torch::jit::Module>(
            torch::jit::load(model_stream, torch::Device(torch::kCPU)));
        module->eval();
        secure_clear(model_string);
        return module;
    } catch (const std::exception& e) {
        secure_clear(model_string);
        throw std::runtime_error(
            std::string("torch::jit::load failed after AES-GCM decrypt; embedded key or blob may be corrupt: ") +
            e.what());
    } catch (...) {
        secure_clear(model_string);
        throw;
    }
}

}  // namespace

extern "C" {

int model_create(ModelHandle* out_handle) {
    if (out_handle == nullptr) {
        return fail(nullptr, -1, "model_create received null out_handle");
    }
    *out_handle = nullptr;

    const auto total_start = Clock::now();
    try {
        auto ctx = std::make_unique<ModelContext>();

        const auto decrypt_start = Clock::now();
        ModelBlob model_blob = load_embedded_model_blob();
        const auto decrypt_end = Clock::now();

        const auto load_start = Clock::now();
        try {
            ctx->model = load_jit_model_from_memory(model_blob.plaintext);
        } catch (...) {
            secure_clear(model_blob.plaintext);
            throw;
        }
        const auto load_end = Clock::now();
        secure_clear(model_blob.plaintext);

        ctx->create_timing.algorithm = model_blob.algorithm;
        ctx->create_timing.encrypted_bytes = model_blob.encrypted_pack_bytes;
        ctx->create_timing.plaintext_bytes = model_blob.plaintext_bytes;
        ctx->create_timing.decrypt_ms = model_blob.encrypted ? elapsed_ms(decrypt_start, decrypt_end) : 0.0;
        ctx->create_timing.jit_load_ms = elapsed_ms(load_start, load_end);
        ctx->create_timing.total_ms = elapsed_ms(total_start, Clock::now());
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
    if (ctx == nullptr || ctx->model == nullptr) {
        return fail(ctx, -1, "model_forward received null handle");
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

        std::vector<torch::jit::IValue> jit_inputs;
        jit_inputs.emplace_back(x);
        torch::Tensor y = ctx->model->forward(jit_inputs).toTensor();
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

int model_last_create_timing(ModelHandle handle, ModelCreateTiming* out_timing) {
    ModelContext* ctx = as_context(handle);
    if (ctx == nullptr || out_timing == nullptr) {
        return -1;
    }
    out_timing->algorithm = ctx->create_timing.algorithm.c_str();
    out_timing->encrypted_bytes = ctx->create_timing.encrypted_bytes;
    out_timing->plaintext_bytes = ctx->create_timing.plaintext_bytes;
    out_timing->decrypt_ms = ctx->create_timing.decrypt_ms;
    out_timing->jit_load_ms = ctx->create_timing.jit_load_ms;
    out_timing->total_ms = ctx->create_timing.total_ms;
    return 0;
}

const char* model_version() {
    return "pt2so-factor-mlp/0.7.0-gcm-aes-128-embedded-obfuscated-key";
}

}
