
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <thread>

#include <gtest/gtest.h>

#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/tools/memory_check.hpp>
#include <mgcpp/cpu_matrix.hpp>
#define private public
#include <mgcpp/gpu_matrix.hpp>

TEST(gpu_matrix, default_constructor)
{
    mgcpp::gpu::matrix<float, 0> mat;

    auto shape = mat.shape();
    EXPECT_EQ(shape.first, 0);
    EXPECT_EQ(shape.second, 0);
    EXPECT_EQ(mat._data, nullptr);
    EXPECT_EQ(mat._context, 
              mgcpp::gpu::matrix<float>()._context);
    EXPECT_TRUE(mat._released);
}

TEST(gpu_matrix, contextless_dimension_constructor)
{
    auto leak_chk = mgcpp::leak_checker();
    {
        size_t row_dim = 10;
        size_t col_dim = 5;
        mgcpp::gpu::matrix<float, 0> mat(row_dim, col_dim);

        auto after = mgcpp::cuda_mem_get_info();
        EXPECT_TRUE(after);
        auto after_memory = after.value().first;

        EXPECT_GT(leak_chk.initial_memory(), after_memory);

        auto shape = mat.shape();
        EXPECT_EQ(shape.first, row_dim);
        EXPECT_EQ(shape.second, col_dim);
        EXPECT_NE(mat._data, nullptr);
        EXPECT_EQ(mat._context, 
                  mgcpp::gpu::matrix<float>()._context);
        EXPECT_FALSE(mat._released);
    }

    EXPECT_TRUE(leak_chk);
}

TEST(gpu_matrix, contextless_dimension_initializing_constructor)
{
    auto leak_chk = mgcpp::leak_checker();
    {
        size_t row_dim = 5;
        size_t col_dim = 10;
        float init_val = 7;
        mgcpp::gpu::matrix<float> mat(row_dim, col_dim, init_val);

        // auto after = mgcpp::cuda_mem_get_info();
        // EXPECT_TRUE(after);
        // auto after_memory = after.value().first;

        // EXPECT_GT(leak_chk.initial_memory(), after_memory);

        // auto shape = mat.shape();
        // EXPECT_EQ(shape.first, row_dim);
        // EXPECT_EQ(shape.second, col_dim);

        // EXPECT_EQ(mat._context, 
        //           mgcpp::gpu::matrix<float>()._context);
        // EXPECT_NE(mat._data, nullptr);
        // EXPECT_FALSE(mat._released);

        // for(size_t i = 0; i < row_dim; ++i)
        // {
        //     for(size_t j = 0; j < col_dim; ++j)
        //     {
        //         EXPECT_EQ(mat.check_value(i, j), init_val)
        //             << "index i: " << i << " j: " << j;
        //     }
        // }
    }

    EXPECT_TRUE(leak_chk);
}

TEST(gpu_matrix, matrix_resize)
{

    auto leak_chk = mgcpp::leak_checker();
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

    EXPECT_TRUE(leak_chk);
}

TEST(gpu_matrix, matrix_zero_after_allocation)
{
    size_t row_dim = 5;
    size_t col_dim = 10;
    mgcpp::gpu::matrix<float> mat(row_dim, col_dim);
    mat.zeros();

    for(size_t i = 0; i < row_dim; ++i)
    {
        for(size_t j = 0; j < col_dim; ++j)
        {

            EXPECT_EQ(mat.check_value(i, j), 0)
                << "index i: " << i << " j: " << j;
        }
    }
}

TEST(gpu_matrix, matrix_zero_without_allocation_failure)
{
    mgcpp::gpu::matrix<float> mat{};

    EXPECT_ANY_THROW(mat.zeros());
}

TEST(gpu_matrix, matrix_resize_init)
{
    auto leak_chk = mgcpp::leak_checker();
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

    EXPECT_TRUE(leak_chk);
}

TEST(gpu_matrix, init_from_cpu_matrix)
{
    size_t row_dim = 5;
    size_t col_dim = 10;
    size_t init_val = 17;

    mgcpp::cpu::matrix<float> cpu_mat(row_dim, col_dim, init_val);

    for(size_t i = 0; i < row_dim; ++i)
    {
        for(size_t j = 0; j < col_dim; ++j)
        {
            EXPECT_EQ(cpu_mat(i, j), init_val)
                << "index i: " << i << " j: " << j;
        }
    }

    auto leak_chk = mgcpp::leak_checker();
    {
        mgcpp::gpu::matrix<float> gpu_mat(cpu_mat);

        for(size_t i = 0; i < row_dim; ++i)
        {
            for(size_t j = 0; j < col_dim; ++j)
            {
                EXPECT_EQ(gpu_mat.check_value(i, j), init_val)
                    << "index i: " << i << " j: " << j;
            }
        }
    }

    EXPECT_TRUE(leak_chk);
}

TEST(gpu_matrix, copy_from_host_matrix)
{
    size_t row_dim = 5;
    size_t col_dim = 10;

    mgcpp::cpu::matrix<float> cpu_mat(row_dim, col_dim);

    size_t counter = 0;
    for(size_t i = 0; i < row_dim; ++i)
    {
        for(size_t j = 0; j < col_dim; ++j)
        {
            cpu_mat(i, j) = counter;
        }
    }

    mgcpp::gpu::matrix<float> gpu_mat(row_dim, col_dim);

    EXPECT_NO_THROW({gpu_mat.copy_from_host(cpu_mat);});
    for(size_t i = 0; i < row_dim; ++i)
    {
        for(size_t j = 0; j < col_dim; ++j)
        {
            EXPECT_EQ(gpu_mat.check_value(i, j), cpu_mat(i, j))
                << "index i: " << i << " j: " << j;
        }
    }
}

TEST(gpu_matrix, copy_to_host)
{
    size_t row_dim = 5;
    size_t col_dim = 10;

    mgcpp::cpu::matrix<float> cpu_mat(row_dim, col_dim);

    size_t counter = 0;
    for(size_t i = 0; i < row_dim; ++i)
    {
        for(size_t j = 0; j < col_dim; ++j)
        {
            cpu_mat(i, j) = counter;
        }
    }

    mgcpp::gpu::matrix<float> gpu_mat(row_dim, col_dim);

    EXPECT_NO_THROW({gpu_mat.copy_from_host(cpu_mat);});
    for(size_t i = 0; i < row_dim; ++i)
    {
        for(size_t j = 0; j < col_dim; ++j)
        {
            EXPECT_EQ(gpu_mat.check_value(i, j), cpu_mat(i, j))
                << "index i: " << i << " j: " << j;
        }
    }
}
