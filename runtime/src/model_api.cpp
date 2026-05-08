#include "model_api.h"

#include "model_session.h"

#include <exception>
#include <memory>
#include <string>

namespace {

thread_local std::string g_last_error;

pt2so::ModelSession* as_session(ModelHandle handle) {
    return reinterpret_cast<pt2so::ModelSession*>(handle);
}

int fail_global(int code, const std::string& message) {
    g_last_error = message;
    return code;
}

}  // namespace

extern "C" {

int model_create(ModelHandle* out_handle) {
    if (out_handle == nullptr) {
        return fail_global(-1, "model_create received null out_handle");
    }
    *out_handle = nullptr;

    try {
        auto session = std::make_unique<pt2so::ModelSession>();
        *out_handle = reinterpret_cast<ModelHandle>(session.release());
        g_last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        return fail_global(-2, e.what());
    } catch (...) {
        return fail_global(-2, "unknown exception in model_create");
    }
}

int model_forward(
    ModelHandle handle,
    const TensorDesc* inputs,
    int input_count,
    TensorDesc* outputs,
    int output_count) {
    pt2so::ModelSession* session = as_session(handle);
    if (session == nullptr) {
        return fail_global(-1, "model_forward received null handle");
    }
    return session->forward_c(inputs, input_count, outputs, output_count);
}

void model_destroy(ModelHandle handle) {
    delete as_session(handle);
}

const char* model_last_error(ModelHandle handle) {
    pt2so::ModelSession* session = as_session(handle);
    if (session == nullptr) {
        return g_last_error.c_str();
    }
    return session->last_error_c_str();
}

int model_last_create_timing(ModelHandle handle, ModelCreateTiming* out_timing) {
    pt2so::ModelSession* session = as_session(handle);
    if (session == nullptr || out_timing == nullptr) {
        return -1;
    }

    const pt2so::ModelSessionCreateTiming& timing = session->create_timing();
    out_timing->algorithm = timing.algorithm.c_str();
    out_timing->encrypted_bytes = timing.encrypted_bytes;
    out_timing->plaintext_bytes = timing.plaintext_bytes;
    out_timing->decrypt_ms = timing.decrypt_ms;
    out_timing->jit_load_ms = timing.jit_load_ms;
    out_timing->total_ms = timing.total_ms;
    return 0;
}

const char* model_version() {
    return "pt2so-factor-mlp/0.7.0-gcm-aes-128-embedded-obfuscated-key";
}

}
