#include "model_session.h"

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

namespace pt2so {
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

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
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

struct ModelSession::Impl {
    std::unique_ptr<torch::jit::Module> model;
    std::string last_error;
    ModelSessionCreateTiming create_timing;
    std::mutex forward_mu;
};

ModelSession::ModelSession() : impl_(std::make_unique<Impl>()) {
    const auto total_start = Clock::now();

    const auto decrypt_start = Clock::now();
    ModelBlob model_blob = load_embedded_model_blob();
    const auto decrypt_end = Clock::now();

    const auto load_start = Clock::now();
    try {
        impl_->model = load_jit_model_from_memory(model_blob.plaintext);
    } catch (...) {
        secure_clear(model_blob.plaintext);
        throw;
    }
    const auto load_end = Clock::now();
    secure_clear(model_blob.plaintext);

    impl_->create_timing.algorithm = model_blob.algorithm;
    impl_->create_timing.encrypted_bytes = model_blob.encrypted_pack_bytes;
    impl_->create_timing.plaintext_bytes = model_blob.plaintext_bytes;
    impl_->create_timing.decrypt_ms = model_blob.encrypted ? elapsed_ms(decrypt_start, decrypt_end) : 0.0;
    impl_->create_timing.jit_load_ms = elapsed_ms(load_start, load_end);
    impl_->create_timing.total_ms = elapsed_ms(total_start, Clock::now());
    impl_->last_error.clear();
}

ModelSession::~ModelSession() = default;
ModelSession::ModelSession(ModelSession&&) noexcept = default;
ModelSession& ModelSession::operator=(ModelSession&&) noexcept = default;

ModelOutput ModelSession::forward(const float* input, int64_t rows, int64_t features) {
    if (impl_ == nullptr || impl_->model == nullptr) {
        throw std::runtime_error("ModelSession has no loaded model");
    }
    if (features != kNumFeatures) {
        std::ostringstream oss;
        oss << "model input feature dimension must be " << kNumFeatures
            << ", got " << features;
        throw std::runtime_error(oss.str());
    }

    const size_t input_bytes = checked_bytes_for_float_tensor(rows, features, "input");
    if (input_bytes > 0 && input == nullptr) {
        throw std::runtime_error("model input data is null");
    }

    std::lock_guard<std::mutex> lock(impl_->forward_mu);
    torch::InferenceMode guard;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor x;
    if (input_bytes == 0) {
        x = torch::empty({rows, kNumFeatures}, options);
    } else {
        x = torch::from_blob(
            const_cast<float*>(input),
            {rows, kNumFeatures},
            options);
    }

    std::vector<torch::jit::IValue> jit_inputs;
    jit_inputs.emplace_back(x);
    torch::Tensor y = impl_->model->forward(jit_inputs).toTensor();
    y = y.to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (y.dim() != 2 || y.size(0) != rows || y.size(1) != 1) {
        throw std::runtime_error("model produced an unexpected output shape");
    }

    ModelOutput output;
    output.rows = y.size(0);
    output.cols = y.size(1);
    const size_t output_bytes = checked_bytes_for_float_tensor(output.rows, output.cols, "output");
    output.data.resize(output_bytes / sizeof(float));
    if (output_bytes != 0) {
        std::memcpy(output.data.data(), y.data_ptr(), output_bytes);
    }
    impl_->last_error.clear();
    return output;
}

ModelOutput ModelSession::forward(const std::vector<float>& input, int64_t rows, int64_t features) {
    const size_t expected_bytes = checked_bytes_for_float_tensor(rows, features, "input");
    if (input.size() != expected_bytes / sizeof(float)) {
        std::ostringstream oss;
        oss << "model input vector size mismatch: need " << (expected_bytes / sizeof(float))
            << " floats, got " << input.size();
        throw std::runtime_error(oss.str());
    }
    return forward(input.empty() ? nullptr : input.data(), rows, features);
}

int ModelSession::forward_c(
    const TensorDesc* inputs,
    int input_count,
    TensorDesc* outputs,
    int output_count) noexcept {
    try {
        if (inputs == nullptr) {
            impl_->last_error = "model_forward received null inputs";
            return -1;
        }
        if (outputs == nullptr) {
            impl_->last_error = "model_forward received null outputs";
            return -1;
        }
        if (input_count != 1) {
            impl_->last_error = "model_forward expects exactly 1 input";
            return -1;
        }
        if (output_count != 1) {
            impl_->last_error = "model_forward expects exactly 1 output";
            return -1;
        }

        const TensorDesc& input = inputs[0];
        TensorDesc& output = outputs[0];

        if (input.dtype != MODEL_FLOAT32) {
            impl_->last_error = "model_forward input dtype must be MODEL_FLOAT32";
            return -1;
        }
        if (input.ndim != 2) {
            impl_->last_error = "model_forward input rank must be 2";
            return -1;
        }
        const int64_t batch = input.shape[0];
        const int64_t features = input.shape[1];
        if (batch < 0) {
            impl_->last_error = "model_forward input batch dimension must be non-negative";
            return -1;
        }
        if (features != kNumFeatures) {
            std::ostringstream oss;
            oss << "model_forward input feature dimension must be " << kNumFeatures
                << ", got " << features;
            impl_->last_error = oss.str();
            return -1;
        }

        const size_t expected_input_bytes = checked_bytes_for_float_tensor(batch, features, "input");
        if (input.bytes < expected_input_bytes) {
            std::ostringstream oss;
            oss << "model_forward input buffer is too small: need " << expected_input_bytes
                << " bytes, got " << input.bytes;
            impl_->last_error = oss.str();
            return -1;
        }
        if (expected_input_bytes > 0 && input.data == nullptr) {
            impl_->last_error = "model_forward input data is null";
            return -1;
        }

        ModelOutput model_output = forward(
            static_cast<const float*>(input.data),
            batch,
            features);

        void* output_data = output.data;
        const size_t output_capacity = output.bytes;
        fill_output_desc_shape(output, model_output.rows);
        const size_t required_output_bytes = output.bytes;
        if ((required_output_bytes > 0 && output_data == nullptr) || output_capacity < required_output_bytes) {
            std::ostringstream oss;
            oss << "model_forward output buffer is null or too small: need "
                << required_output_bytes << " bytes, got " << output_capacity;
            impl_->last_error = oss.str();
            return -3;
        }

        if (required_output_bytes != 0) {
            std::memcpy(output_data, model_output.data.data(), required_output_bytes);
        }
        output.bytes = required_output_bytes;
        impl_->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        impl_->last_error = e.what();
        return -2;
    } catch (...) {
        impl_->last_error = "unknown exception in model_forward";
        return -2;
    }
}

const ModelSessionCreateTiming& ModelSession::create_timing() const noexcept {
    return impl_->create_timing;
}

const std::string& ModelSession::last_error() const noexcept {
    return impl_->last_error;
}

const char* ModelSession::last_error_c_str() const noexcept {
    return impl_->last_error.c_str();
}

}  // namespace pt2so
