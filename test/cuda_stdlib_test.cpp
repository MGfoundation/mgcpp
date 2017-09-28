
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <mgcpp/cuda/stdlib.hpp>

TEST(cuda_malloc, cuda_malloc_success)
{
    float* ptr = nullptr;
    (void)ptr; // warning suppression

    size_t free_memory_before_malloc = 0;
    cudaMemGetInfo(&free_memory_before_malloc, nullptr);

    EXPECT_NO_THROW({ptr = mgcpp::cuda_malloc<float>(10);});
    EXPECT_NE(ptr, nullptr);

    size_t free_memory_after_malloc = 0;
    cudaMemGetInfo(&free_memory_after_malloc, nullptr);

    EXPECT_GT(free_memory_before_malloc, free_memory_after_malloc);

    mgcpp::cuda_free(ptr);
}

TEST(cuda_malloc, cuda_malloc_throw_failure)
{
    float* ptr = nullptr;
    (void)ptr; // warning suppression

    size_t free_memory= 0;
    cudaMemGetInfo(&free_memory, nullptr);

    EXPECT_ANY_THROW(
        {ptr = mgcpp::cuda_malloc<float>(free_memory * 2);});
}

TEST(cuda_malloc, cuda_malloc_nothrow_failure)
{
    float* ptr = nullptr;
    (void)ptr; // warning suppression

    size_t free_memory= 0;
    cudaMemGetInfo(&free_memory, nullptr);

    EXPECT_NO_THROW(
        {
            ptr = mgcpp::cuda_malloc<float>(free_memory * 2,
                                            std::nothrow);
        });

    EXPECT_EQ(ptr, nullptr);
}

TEST(cuda_malloc, cuda_malloc_nothrow_success)
{
    size_t free_memory_before_malloc = 0;
    cudaMemGetInfo(&free_memory_before_malloc, nullptr);

    float* ptr = nullptr;
    ptr = mgcpp::cuda_malloc<float>(10, std::nothrow);
    EXPECT_NE(ptr, nullptr);

    size_t free_memory_after_malloc = 0;
    cudaMemGetInfo(&free_memory_after_malloc, nullptr);

    EXPECT_GT(free_memory_before_malloc, free_memory_after_malloc);

    mgcpp::cuda_free(ptr);
}

TEST(cuda_free, cuda_free_success)
{
    size_t free_memory_before_malloc = 0;
    cudaMemGetInfo(&free_memory_before_malloc, nullptr);

    float* ptr = nullptr;

    EXPECT_NO_THROW({ptr = mgcpp::cuda_malloc<float>(10);});
    EXPECT_NE(ptr, nullptr);

    size_t free_memory_after_malloc = 0;
    cudaMemGetInfo(&free_memory_after_malloc, nullptr);

    EXPECT_GT(free_memory_before_malloc, free_memory_after_malloc);

    bool success = mgcpp::cuda_free(ptr);
    EXPECT_EQ(success, true);

    size_t free_memory_after_free = 0;
    cudaMemGetInfo(&free_memory_after_free, nullptr);

    EXPECT_EQ(free_memory_after_free, free_memory_before_malloc);
}

TEST(cuda_free, cuda_free_failure)
{
    float* ptr = (float*)10u;
    bool success = mgcpp::cuda_free(ptr);
    EXPECT_EQ(success, false);
}
