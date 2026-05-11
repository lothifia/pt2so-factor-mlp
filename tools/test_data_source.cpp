#include "../runtime/include/data_source.h"

#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

template <typename Fn>
void expect_throws(Fn&& fn) {
    bool thrown = false;
    try {
        fn();
    } catch (const std::exception&) {
        thrown = true;
    }
    assert(thrown);
}

void expect_vec_eq(const std::vector<float>& actual, const std::vector<float>& expected) {
    assert(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(actual[i] == expected[i]);
    }
}

void test_accumulates_rows_until_target_batch_ready() {
    pt2so::DataSource<float> ds(/*model_hidden=*/3, /*target_rows=*/4);

    const size_t inserted = ds.insert({
        {0.0f, 1.0f},
        {10.0f, 11.0f},
        {20.0f, 21.0f},
    });
    assert(inserted == 2);
    assert(!ds.try_get_batch().has_value());

    ds.insert({
        {2.0f, 3.0f, 4.0f},
        {12.0f, 13.0f, 14.0f},
        {22.0f, 23.0f, 24.0f},
    });

    auto batch = ds.try_get_batch();
    assert(batch.has_value());
    assert(batch->rows == 4);
    assert(batch->hidden == 3);
    assert(!batch->is_last);
    expect_vec_eq(batch->data, {
        0.0f, 10.0f, 20.0f,
        1.0f, 11.0f, 21.0f,
        2.0f, 12.0f, 22.0f,
        3.0f, 13.0f, 23.0f,
    });

    assert(!ds.try_get_batch().has_value());

    ds.insert({
        {5.0f, 6.0f, 7.0f},
        {15.0f, 16.0f, 17.0f},
        {25.0f, 26.0f, 27.0f},
    });

    auto second = ds.try_get_batch();
    assert(second.has_value());
    assert(second->rows == 4);
    expect_vec_eq(second->data, {
        4.0f, 14.0f, 24.0f,
        5.0f, 15.0f, 25.0f,
        6.0f, 16.0f, 26.0f,
        7.0f, 17.0f, 27.0f,
    });
}

void test_insert_can_exceed_target_rows() {
    pt2so::DataSource<float> ds(/*model_hidden=*/2, /*target_rows=*/3);

    ds.insert({
        {0.0f, 1.0f, 2.0f, 3.0f, 4.0f},
        {10.0f, 11.0f, 12.0f, 13.0f, 14.0f},
    });

    auto first = ds.try_get_batch();
    assert(first.has_value());
    assert(first->rows == 3);
    assert(!first->is_last);
    expect_vec_eq(first->data, {
        0.0f, 10.0f,
        1.0f, 11.0f,
        2.0f, 12.0f,
    });

    assert(!ds.try_get_batch().has_value());

    ds.close();
    auto last = ds.try_get_batch();
    assert(last.has_value());
    assert(last->rows == 2);
    assert(last->is_last);
    expect_vec_eq(last->data, {
        3.0f, 13.0f,
        4.0f, 14.0f,
    });
}

void test_get_batch_async_waits_for_target_rows() {
    pt2so::DataSource<float> ds(/*model_hidden=*/2, /*target_rows=*/3);

    auto fut = ds.get_batch_async();
    assert(fut.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);

    ds.insert({
        {1.0f, 2.0f},
        {4.0f, 5.0f},
    });
    assert(fut.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);

    ds.insert({
        {3.0f},
        {6.0f},
    });

    auto batch = fut.get();
    assert(batch.rows == 3);
    assert(batch.hidden == 2);
    assert(!batch.is_last);
    expect_vec_eq(batch.data, {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f,
    });
}

void test_close_returns_partial_last_batch() {
    pt2so::DataSource<float> ds(/*model_hidden=*/2, /*target_rows=*/4);

    ds.insert({
        {1.0f, 2.0f},
        {3.0f, 4.0f},
    });
    assert(!ds.try_get_batch().has_value());

    ds.close();
    assert(ds.closed());

    auto batch = ds.try_get_batch();
    assert(batch.has_value());
    assert(batch->rows == 2);
    assert(batch->hidden == 2);
    assert(batch->is_last);
    expect_vec_eq(batch->data, {
        1.0f, 3.0f,
        2.0f, 4.0f,
    });

    assert(!ds.try_get_batch().has_value());
}

void test_close_marks_exact_final_batch() {
    pt2so::DataSource<float> ds(/*model_hidden=*/2, /*target_rows=*/2);

    ds.insert({
        {1.0f, 2.0f},
        {3.0f, 4.0f},
    });
    ds.close();

    auto batch = ds.try_get_batch();
    assert(batch.has_value());
    assert(batch->rows == 2);
    assert(batch->is_last);
    expect_vec_eq(batch->data, {
        1.0f, 3.0f,
        2.0f, 4.0f,
    });
}

void test_close_wakes_async_waiter() {
    pt2so::DataSource<float> ds(/*model_hidden=*/2, /*target_rows=*/4);

    auto fut = ds.get_batch_async();
    ds.insert({
        {1.0f},
        {2.0f},
    });
    ds.close();

    auto batch = fut.get();
    assert(batch.rows == 1);
    assert(batch.hidden == 2);
    assert(batch.is_last);
    expect_vec_eq(batch.data, {
        1.0f, 2.0f,
    });
}

void test_close_without_pending_rows_makes_async_throw() {
    pt2so::DataSource<float> ds(/*model_hidden=*/2, /*target_rows=*/4);

    auto fut = ds.get_batch_async();
    ds.close();

    expect_throws([&] {
        (void)fut.get();
    });
}

void test_invalid_inputs_throw() {
    expect_throws([] {
        pt2so::DataSource<float> ds(0, 1);
    });
    expect_throws([] {
        pt2so::DataSource<float> ds(1, 0);
    });

    pt2so::DataSource<float> ds(/*model_hidden=*/2, /*target_rows=*/3);

    expect_throws([&] {
        ds.insert({{1.0f}});
    });
    expect_throws([&] {
        ds.insert({{1.0f, 2.0f}, {3.0f}});
    });

    ds.close();
    expect_throws([&] {
        ds.insert({{1.0f}, {2.0f}});
    });
}

}  // namespace

int main() {
    test_accumulates_rows_until_target_batch_ready();
    test_insert_can_exceed_target_rows();
    test_get_batch_async_waits_for_target_rows();
    test_close_returns_partial_last_batch();
    test_close_marks_exact_final_batch();
    test_close_wakes_async_waiter();
    test_close_without_pending_rows_makes_async_throw();
    test_invalid_inputs_throw();

    std::cout << "test_data_source: OK\n";
    return 0;
}
