#include "model_api.h"

#include <dlfcn.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
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

struct Api {
    int (*model_create)(ModelHandle*){nullptr};
    void (*model_destroy)(ModelHandle){nullptr};
    const char* (*model_last_error)(ModelHandle){nullptr};
    int (*model_last_create_timing)(ModelHandle, ModelCreateTiming*){nullptr};
};

struct TimingRow {
    std::string algorithm;
    size_t encrypted_bytes{0};
    size_t plaintext_bytes{0};
    double decrypt_ms{0.0};
    double jit_load_ms{0.0};
    double total_ms{0.0};
};

std::string last_error(const Api& api, ModelHandle handle) {
    const char* error = api.model_last_error ? api.model_last_error(handle) : nullptr;
    return error ? std::string(error) : std::string();
}

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --lib build/libmodel.so"
        << " [--repeat 5] [--warmup 1]\n\n"
        << "Prints CSV rows:\n"
        << "  algorithm,iteration,encrypted_bytes,plaintext_bytes,decrypt_ms,jit_load_ms,total_create_inner_ms,total_create_wall_ms\n";
}

int parse_nonnegative_int(const std::string& value, const char* name) {
    size_t parsed = 0;
    int result = 0;
    try {
        result = std::stoi(value, &parsed);
    } catch (...) {
        throw std::runtime_error(std::string(name) + " must be an integer");
    }
    if (parsed != value.size() || result < 0) {
        throw std::runtime_error(std::string(name) + " must be a non-negative integer");
    }
    return result;
}

TimingRow create_once(const Api& api, double& wall_ms) {
    ModelHandle handle = nullptr;
    const auto start = Clock::now();
    const int rc = api.model_create(&handle);
    const auto end = Clock::now();
    wall_ms = elapsed_ms(start, end);

    if (rc != 0) {
        throw std::runtime_error("model_create failed rc=" + std::to_string(rc) + ": " + last_error(api, handle));
    }
    if (handle == nullptr) {
        throw std::runtime_error("model_create succeeded but returned a null handle");
    }

    ModelCreateTiming timing{};
    try {
        const int timing_rc = api.model_last_create_timing(handle, &timing);
        if (timing_rc != 0) {
            throw std::runtime_error("model_last_create_timing failed rc=" + std::to_string(timing_rc));
        }
        TimingRow row;
        row.algorithm = timing.algorithm ? timing.algorithm : "unknown";
        row.encrypted_bytes = timing.encrypted_bytes;
        row.plaintext_bytes = timing.plaintext_bytes;
        row.decrypt_ms = timing.decrypt_ms;
        row.jit_load_ms = timing.jit_load_ms;
        row.total_ms = timing.total_ms;
        api.model_destroy(handle);
        return row;
    } catch (...) {
        api.model_destroy(handle);
        throw;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string lib_path = "build/libmodel.so";
    int repeat = 5;
    int warmup = 1;
    void* library = nullptr;

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
            } else if (arg == "--repeat") {
                repeat = parse_nonnegative_int(require_value("--repeat"), "--repeat");
            } else if (arg == "--warmup") {
                warmup = parse_nonnegative_int(require_value("--warmup"), "--warmup");
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        if (repeat <= 0) {
            throw std::runtime_error("--repeat must be greater than zero");
        }

        library = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) {
            const char* error = dlerror();
            throw std::runtime_error(std::string("dlopen failed: ") + (error ? error : "unknown error"));
        }

        Api api{};
        api.model_create = load_symbol<int (*)(ModelHandle*)>(library, "model_create");
        api.model_destroy = load_symbol<void (*)(ModelHandle)>(library, "model_destroy");
        api.model_last_error = load_symbol<const char* (*)(ModelHandle)>(library, "model_last_error");
        api.model_last_create_timing =
            load_symbol<int (*)(ModelHandle, ModelCreateTiming*)>(library, "model_last_create_timing");

        for (int i = 0; i < warmup; ++i) {
            double wall_ms = 0.0;
            (void)create_once(api, wall_ms);
        }

        for (int i = 0; i < repeat; ++i) {
            double wall_ms = 0.0;
            const TimingRow timing = create_once(api, wall_ms);
            std::cout
                << timing.algorithm << ","
                << i << ","
                << timing.encrypted_bytes << ","
                << timing.plaintext_bytes << ","
                << timing.decrypt_ms << ","
                << timing.jit_load_ms << ","
                << timing.total_ms << ","
                << wall_ms << "\n";
        }

        dlclose(library);
        library = nullptr;
        return 0;
    } catch (const std::exception& e) {
        if (library != nullptr) {
            dlclose(library);
        }
        std::cerr << "error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
}
