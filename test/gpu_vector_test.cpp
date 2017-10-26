
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#define ERROR_CHECK_EXCEPTION true

#define private public
#include <mgcpp/host/vector.hpp>
#include <mgcpp/device/vector.hpp>

TEST(gpu_vector, default_constructor)
{
    mgcpp::device_vector<float> vec{};

    auto shape = vec.shape();
    EXPECT_EQ(shape, 0);
    EXPECT_EQ(vec._data, nullptr);
    EXPECT_EQ(vec._context, 
              mgcpp::device_vector<float>()._context);
    EXPECT_TRUE(vec._released);
}

TEST(gpu_vector, size_constructor)
{
    size_t size = 10;
    mgcpp::device_vector<float> vec(size);

    auto shape = vec.shape();
    EXPECT_EQ(shape, 10);
    EXPECT_NE(vec._data, nullptr);
    EXPECT_EQ(vec._context, 
              mgcpp::device_vector<float>()._context);
    EXPECT_FALSE(vec._released);
}

TEST(gpu_vector, initializing_constructor)
{
    size_t size = 10;
    float init_val = 7;

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    mgcpp::device_vector<float> vec(size, init_val);

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_GT(before_memory, after_memory);

    auto shape = vec.shape();
    EXPECT_EQ(shape, 10);
    EXPECT_NE(vec._data, nullptr);
    EXPECT_EQ(vec._context, 
              mgcpp::device_vector<float>()._context);
    EXPECT_FALSE(vec._released);

    for(auto i = 0u; i < size; ++i)
    {
        EXPECT_EQ(vec.check_value(i), init_val);
    }
}

TEST(gpu_vector, cpy_constructor)
{
    size_t size = 10;
    size_t init_val = 7;
    mgcpp::device_vector<float> original(size, init_val);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    mgcpp::device_vector<float> copied(original);

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_GT(before_memory, after_memory);

    auto shape = copied.shape();
    EXPECT_EQ(shape, 10);
    EXPECT_NE(copied._data, nullptr);
    EXPECT_EQ(copied._context, 
              mgcpp::device_vector<float>()._context);
    EXPECT_FALSE(copied._released);

    for(auto i = 0u; i < size; ++i)
    {
        EXPECT_EQ(copied.check_value(i), init_val);
    }
}

TEST(gpu_vector, move_constructor)
{
    size_t size = 10;
    size_t init_val = 7;
    mgcpp::device_vector<float> original(size, init_val);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    mgcpp::device_vector<float> moved(std::move(original));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_EQ(before_memory, after_memory);

    auto shape = moved.shape();
    EXPECT_EQ(shape, 10);
    EXPECT_NE(moved._data, nullptr);
    EXPECT_EQ(moved._context, 
              mgcpp::device_vector<float>()._context);
    EXPECT_FALSE(moved._released);

    for(auto i = 0u; i < size; ++i)
    {
        EXPECT_EQ(moved.check_value(i), init_val);
    }

    auto ori_shape = original.shape();
    EXPECT_EQ(ori_shape, 0);
    EXPECT_EQ(original._data, nullptr);
    EXPECT_TRUE(original._released);
}

TEST(gpu_vector, cpy_assign_operator)
{
    size_t size = 10;
    size_t init_val = 7;
    mgcpp::device_vector<float> original(size, init_val);
    mgcpp::device_vector<float> copied(size * 3, init_val);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    copied = original;

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_LT(before_memory, after_memory);

    auto shape = copied.shape();
    EXPECT_EQ(shape, 10);
    EXPECT_NE(copied._data, nullptr);
    EXPECT_EQ(copied._context, 
              mgcpp::device_vector<float>()._context);
    EXPECT_FALSE(copied._released);

    for(auto i = 0u; i < size; ++i)
    {
        EXPECT_EQ(copied.check_value(i), init_val);
    }
}

TEST(gpu_vector, move_assign_operator)
{
    size_t size = 10;
    size_t init_val = 7;
    mgcpp::device_vector<float> original(size, init_val);
    mgcpp::device_vector<float> moved(size, init_val);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    moved = std::move(original);

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_LT(before_memory, after_memory);

    auto shape = moved.shape();
    EXPECT_EQ(shape, 10);
    EXPECT_NE(moved._data, nullptr);
    EXPECT_EQ(moved._context, 
              mgcpp::device_vector<float>()._context);
    EXPECT_FALSE(moved._released);

    for(auto i = 0u; i < size; ++i)
    {
        EXPECT_EQ(moved.check_value(i), init_val);
    }

    auto ori_shape = original.shape();
    EXPECT_EQ(ori_shape, 0);
    EXPECT_EQ(original._data, nullptr);
    EXPECT_TRUE(original._released);
}
