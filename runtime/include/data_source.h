#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pt2so {

template <typename T>
struct InputBatch {
    std::vector<T> data; // row-major [rows, hidden]
    int64_t rows{0};
    int64_t hidden{0};
    bool is_last{false};
};

template <typename T>
class DataSource {
public:
    DataSource(int64_t model_hidden, int64_t target_rows)
        : state_(std::make_shared<State>(model_hidden, target_rows)) {}

    ~DataSource() {
        close();
    }

    DataSource(const DataSource&) = delete;
    DataSource& operator=(const DataSource&) = delete;
    DataSource(DataSource&& other) noexcept : state_(std::move(other.state_)) {}

    DataSource& operator=(DataSource&& other) noexcept {
        if (this != &other) {
            close();
            state_ = std::move(other.state_);
        }
        return *this;
    }

    size_t insert(std::vector<std::vector<T>> rows) {
        if (rows.empty()) {
            return 0;
        }

        auto state = state_;
        if (!state) {
            throw std::runtime_error("DataSource has no state");
        }
        const int64_t rows_count = static_cast<int64_t>(rows.size());
        const int64_t hidden = state->model_hidden;

        std::vector<T> flat;
        flat.reserve(checked_element_count(rows_count, hidden));
        for (auto& row : rows) {
            if (static_cast<int64_t>(row.size()) != hidden) {
                throw std::runtime_error("DataSource::insert expects rows shaped [rows, hidden]");
            }
            flat.insert(
                flat.end(),
                std::make_move_iterator(row.begin()),
                std::make_move_iterator(row.end()));
        }

        bool should_notify = false;
        {
            std::lock_guard<std::mutex> lock(state->mu);
            if (state->closed) {
                throw std::runtime_error("DataSource::insert called after close");
            }

            state->chunks.push_back(RowChunk{
                std::move(flat),
                rows_count,
                0,
            });
            state->rows_ready += rows_count;
            should_notify = state->rows_ready >= state->target_rows;
        }

        if (should_notify) {
            state->cv.notify_all();
        }
        return static_cast<size_t>(rows_count);
    }

    std::optional<InputBatch<T>> try_get_batch() {
        auto state = state_;
        if (!state) {
            return std::nullopt;
        }
        std::lock_guard<std::mutex> lock(state->mu);
        return try_get_batch_locked(*state);
    }

    std::future<InputBatch<T>> get_batch_async() {
        auto state = state_;
        if (!state) {
            return std::async(std::launch::deferred, []() -> InputBatch<T> {
                throw std::runtime_error("DataSource has no state");
            });
        }
        return std::async(std::launch::async, [state] {
            std::unique_lock<std::mutex> lock(state->mu);
            state->cv.wait(lock, [&] {
                return state->rows_ready >= state->target_rows || state->closed;
            });

            auto batch = try_get_batch_locked(*state);
            if (!batch.has_value()) {
                throw std::runtime_error("DataSource closed with no pending rows");
            }
            return std::move(*batch);
        });
    }

    void close() {
        auto state = state_;
        if (!state) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(state->mu);
            if (state->closed) {
                return;
            }
            state->closed = true;
        }
        state->cv.notify_all();
    }

    bool closed() const {
        auto state = state_;
        if (!state) {
            return true;
        }
        std::lock_guard<std::mutex> lock(state->mu);
        return state->closed;
    }

private:
    struct RowChunk {
        std::vector<T> data; // row-major [rows, hidden]
        int64_t rows{0};
        int64_t offset_rows{0};
    };

    struct State {
        State(int64_t model_hidden_value, int64_t target_rows_value)
            : model_hidden(model_hidden_value), target_rows(target_rows_value) {
            if (model_hidden <= 0) {
                throw std::invalid_argument("model_hidden must be positive");
            }
            if (target_rows <= 0) {
                throw std::invalid_argument("target_rows must be positive");
            }
        }

        int64_t model_hidden;
        int64_t target_rows;
        int64_t rows_ready{0};
        bool closed{false};

        std::deque<RowChunk> chunks;
        mutable std::mutex mu;
        std::condition_variable cv;
    };

    static size_t checked_element_count(int64_t rows, int64_t hidden) {
        if (rows < 0 || hidden < 0) {
            throw std::runtime_error("DataSource shape contains a negative dimension");
        }
        const auto urows = static_cast<size_t>(rows);
        const auto uhidden = static_cast<size_t>(hidden);
        if (uhidden != 0 && urows > std::numeric_limits<size_t>::max() / uhidden) {
            throw std::runtime_error("DataSource element count overflows size_t");
        }
        return urows * uhidden;
    }

    static std::optional<InputBatch<T>> try_get_batch_locked(State& state) {
        if (state.rows_ready >= state.target_rows) {
            const bool is_last = state.closed && state.rows_ready == state.target_rows;
            return take_batch_locked(state, state.target_rows, is_last);
        }
        if (state.closed && state.rows_ready > 0) {
            return take_batch_locked(state, state.rows_ready, true);
        }
        return std::nullopt;
    }

    static InputBatch<T> take_batch_locked(State& state, int64_t rows_to_take, bool is_last) {
        InputBatch<T> batch;
        batch.rows = rows_to_take;
        batch.hidden = state.model_hidden;
        batch.is_last = is_last;
        batch.data.resize(checked_element_count(rows_to_take, state.model_hidden));

        int64_t copied_rows = 0;
        while (copied_rows < rows_to_take) {
            RowChunk& chunk = state.chunks.front();
            const int64_t available_rows = chunk.rows - chunk.offset_rows;
            const int64_t take_rows = std::min(rows_to_take - copied_rows, available_rows);

            auto src_begin = chunk.data.begin() + checked_element_count(chunk.offset_rows, state.model_hidden);
            auto src_end = src_begin + checked_element_count(take_rows, state.model_hidden);
            auto dst_begin = batch.data.begin() + checked_element_count(copied_rows, state.model_hidden);
            std::move(src_begin, src_end, dst_begin);

            chunk.offset_rows += take_rows;
            copied_rows += take_rows;
            state.rows_ready -= take_rows;

            if (chunk.offset_rows == chunk.rows) {
                state.chunks.pop_front();
            }
        }

        return batch;
    }

    std::shared_ptr<State> state_;
};

}  // namespace pt2so
