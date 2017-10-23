
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <thread>

#include <gtest/gtest.h>

#include <mgcpp/cuda/memory.hpp>
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

TEST(gpu_matrix, dimension_constructor)
{
    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    size_t row_dim = 10;
    size_t col_dim = 5;
    mgcpp::gpu::matrix<float, 0> mat(row_dim, col_dim);

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE( after);
    auto after_memory = after.value().first;

    EXPECT_GT(before_memory, after_memory);

    auto shape = mat.shape();
    EXPECT_EQ(shape.first, row_dim);
    EXPECT_EQ(shape.second, col_dim);
    EXPECT_NE(mat._data, nullptr);
    EXPECT_FALSE(mat._released);
}

TEST(gpu_matrix, dimension_initializing_constructor)
{
    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_memory = before.value().first;

    size_t row_dim = 5;
    size_t col_dim = 10;
    float init_val = 7;
    mgcpp::gpu::matrix<float> mat(row_dim, col_dim, init_val);

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_memory = after.value().first;

    EXPECT_GT(before_memory, after_memory);

    auto shape = mat.shape();
    EXPECT_EQ(shape.first, row_dim);
    EXPECT_EQ(shape.second, col_dim);
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

TEST(gpu_matrix, matrix_resize)
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

TEST(gpu_matrix, copy_construction)
{
    size_t row_dim = 5;
    size_t col_dim = 10;
    float init = 7;

    mgcpp::gpu::matrix<float> original(row_dim, col_dim, init);

    mgcpp::gpu::matrix<float> copied(original);

    for(auto i = 0u; i < 5; ++i)
    {
        for(auto j = 0u; j < 5; ++j)
        {
            EXPECT_EQ(original.check_value(i, j),
                      copied.check_value(i, j));
        }
    }

    EXPECT_FALSE(original._released);
    EXPECT_FALSE(copied._released);
    EXPECT_EQ(original._m_dim, copied._m_dim);
    EXPECT_EQ(original._n_dim, copied._n_dim);
}

TEST(gpu_matrix, copy_assign_operator)
{
    size_t row_dim = 5;
    size_t col_dim = 10;
    float init = 7;

    mgcpp::gpu::matrix<float> original(row_dim, col_dim, init);
    mgcpp::gpu::matrix<float> copied(row_dim * 3, col_dim * 3);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_freemem = before.value().first;

    copied = original;

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_freemem = after.value().first;

    EXPECT_LT(before_freemem, after_freemem);

    for(auto i = 0u; i < 5; ++i)
    {
        for(auto j = 0u; j < 5; ++j)
        {
            EXPECT_EQ(original.check_value(i, j),
                      copied.check_value(i, j));
        }
    }

    EXPECT_FALSE(original._released);
    EXPECT_FALSE(copied._released);
    EXPECT_EQ(original._m_dim, copied._m_dim);
    EXPECT_EQ(original._n_dim, copied._n_dim);
}

TEST(gpu_matrix, move_constructor)
{
    size_t row_dim = 5;
    size_t col_dim = 10;
    float init = 7;

    mgcpp::gpu::matrix<float> original(row_dim, col_dim, init);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_freemem = before.value().first;

    mgcpp::gpu::matrix<float> moved(std::move(original));

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_freemem = after.value().first;

    EXPECT_EQ(before_freemem, after_freemem);

    for(auto i = 0u; i < 5; ++i)
    {
        for(auto j = 0u; j < 5; ++j)
        {
            EXPECT_EQ(moved.check_value(i, j), 7);
        }
    }

    EXPECT_TRUE(original._released);
    EXPECT_FALSE(moved._released);
    EXPECT_EQ(moved._m_dim, row_dim);
    EXPECT_EQ(moved._n_dim, col_dim);
}

TEST(gpu_matrix, move_assign_operator)
{
    size_t row_dim = 5;
    size_t col_dim = 10;
    float init = 7;

    mgcpp::gpu::matrix<float> original(row_dim, col_dim, init);
    mgcpp::gpu::matrix<float> moved(row_dim * 2, col_dim * 2);

    auto before = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(before);
    auto before_freemem = before.value().first;


    moved = std::move(original);

    auto after = mgcpp::cuda_mem_get_info();
    EXPECT_TRUE(after);
    auto after_freemem = after.value().first;

    EXPECT_LT(before_freemem, after_freemem);

    for(auto i = 0u; i < 5; ++i)
    {
        for(auto j = 0u; j < 5; ++j)
        {
            EXPECT_EQ(moved.check_value(i, j), 7);
        }
    }

    EXPECT_TRUE(original._released);
    EXPECT_FALSE(moved._released);
    EXPECT_EQ(moved._m_dim, row_dim);
    EXPECT_EQ(moved._n_dim, col_dim);
}
