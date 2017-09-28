
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <mgcpp/cuda/stdlib.hpp>

TEST(cuda_malloc, cuda_malloc_success)
{
    size_t free_memory_before_malloc = 0;
    cudaMemGetInfo(&free_memory_before_malloc, nullptr);

    auto rst = mgcpp::cuda_malloc<float>(10);
    EXPECT_TRUE(rst);

    size_t free_memory_after_malloc = 0;
    cudaMemGetInfo(&free_memory_after_malloc, nullptr);

    EXPECT_GT(free_memory_before_malloc, free_memory_after_malloc);

    (void)mgcpp::cuda_free(rst.value());
}

TEST(cuda_malloc, cuda_malloc_failure)
{
    size_t free_memory= 0;
    cudaMemGetInfo(&free_memory, nullptr);

    auto ptr = mgcpp::cuda_malloc<float>(free_memory * 2);
    EXPECT_FALSE(ptr);
}

TEST(cuda_free, cuda_free_success)
{
    size_t free_memory_before_malloc = 0;
    cudaMemGetInfo(&free_memory_before_malloc, nullptr);

    auto result = mgcpp::cuda_malloc<float>(10);;
    EXPECT_TRUE(result);

    size_t free_memory_after_malloc = 0;
    cudaMemGetInfo(&free_memory_after_malloc, nullptr);

    EXPECT_GT(free_memory_before_malloc, free_memory_after_malloc);

    auto free_result = mgcpp::cuda_free(result.value());
    EXPECT_TRUE(free_result); 

    size_t free_memory_after_free = 0;
    cudaMemGetInfo(&free_memory_after_free, nullptr);

    EXPECT_EQ(free_memory_after_free, free_memory_before_malloc);
}

TEST(cuda_free, cuda_free_failure)
{
    float* ptr = (float*)10u;
    auto result = mgcpp::cuda_free(ptr);
    EXPECT_FALSE(result);
}
