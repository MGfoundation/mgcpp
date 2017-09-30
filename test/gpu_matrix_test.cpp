
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <cstdio>

#include <gtest/gtest.h>

#include <mgcpp/cuda/memory.hpp>
#define private public
#include <mgcpp/gpu_matrix.hpp>

TEST(gpu_matrix, default_constructor)
{
    mgcpp::gpu::matrix<float, 0> mat;

    EXPECT_EQ(mat.rows(), 0u);
    EXPECT_EQ(mat.columns(), 0u);
    EXPECT_EQ(mat._data, nullptr);
    EXPECT_EQ(mat._context, nullptr);
    EXPECT_TRUE(mat._released);
}

TEST(gpu_matrix, contextless_dimension_constructor)
{
    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;
    
    {
        size_t row_dim = 10;
        size_t col_dim = 10;
        mgcpp::gpu::matrix<float, 0> mat(row_dim, col_dim);

        auto after = mgcpp::cuda_mem_get_info();
        EXPECT_TRUE(after);
        auto after_memory = after.value().first;

        EXPECT_GT(before_memory, after_memory);

        EXPECT_EQ(mat.rows(), row_dim);
        EXPECT_EQ(mat.columns(), col_dim);
        EXPECT_NE(mat._data, nullptr);
        EXPECT_EQ(mat._context, nullptr);
        EXPECT_FALSE(mat._released);
    }

    auto last = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(last);
    auto last_memory = last.value().first;

    EXPECT_EQ(before_memory, last_memory);
}

TEST(gpu_matrix, contextless_dimension_initializing_constructor)
{
    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    {
        size_t row_dim = 10;
        size_t col_dim = 10;
        float init_val = 10;
        mgcpp::gpu::matrix<float> mat(row_dim, col_dim, init_val);

        auto after = mgcpp::cuda_mem_get_info();
        EXPECT_TRUE(after);
        auto after_memory = after.value().first;

        EXPECT_GT(before_memory, after_memory);

        EXPECT_EQ(mat.rows(), row_dim);
        EXPECT_EQ(mat.columns(), col_dim);
        EXPECT_EQ(mat._context, nullptr);
        EXPECT_NE(mat._data, nullptr);
        EXPECT_FALSE(mat._released);

        for(size_t i = 0; i < row_dim; ++i)
        {
            for(size_t j = 0; j < col_dim; ++j)
            {
                EXPECT_EQ(mat.check_value(i, j), init_val)
                    << "index i: " << i << " j: " << j;
            }
        }
    }

    auto last = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(last);
    auto last_memory = last.value().first;

    EXPECT_EQ(before_memory, last_memory);
}

TEST(gpu_matrix, matrix_resize)
{
    auto initial = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(initial);
    auto initial_freemem = initial.value().first;

    {
        auto tenbyten = mgcpp::cuda_mem_get_info();
        EXPECT_TRUE(tenbyten);
        auto tenbyten_freemem = tenbyten.value().first;

        size_t row_dim = 10;
        size_t col_dim = 10;
        mgcpp::gpu::matrix<float> mat(row_dim, col_dim);

        mat.resize(100, 100);

        auto kilobykilo = mgcpp::cuda_mem_get_info();
        EXPECT_TRUE(tenbyten);
        auto kilobykilo_freemem = kilobykilo.value().first;

        EXPECT_GT(tenbyten_freemem, kilobykilo_freemem);
    }

    auto finally = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(finally);
    auto final_freemem = initial.value().first;
    EXPECT_EQ(initial_freemem, final_freemem);
}

TEST(gpu_matrix, matrix_resize_init)
{
    auto initial = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(initial);
    auto initial_freemem = initial.value().first;

    {
        auto twobytwo = mgcpp::cuda_mem_get_info();
        EXPECT_TRUE(twobytwo);
        auto twobytwo_freemem = twobytwo.value().first;

        mgcpp::gpu::matrix<float> mat(2, 2);

        size_t row_dim = 5;
        size_t col_dim = 5;
        size_t init_val = 17;
        mat.resize(row_dim, col_dim, init_val);

        for(size_t i = 0; i < row_dim; ++i)
        {
            for(size_t j = 0; j < col_dim; ++j)
            {
                EXPECT_EQ(mat.check_value(i, j), init_val)
                    << "index i: " << i << " j: " << j;
            }
        }

        auto fivebyfive = mgcpp::cuda_mem_get_info();
        EXPECT_TRUE(fivebyfive);
        auto fivebyfive_freemem = fivebyfive.value().first;

        EXPECT_GT(twobytwo_freemem, fivebyfive_freemem);
    }

    auto finally = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(finally);
    auto final_freemem = initial.value().first;
    EXPECT_EQ(initial_freemem, final_freemem);
}
