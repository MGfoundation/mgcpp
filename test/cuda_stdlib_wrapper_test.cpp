
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/cuda/internal/stdlib_wrapper.hpp>
#include <mgcpp/cuda/internal/status_wrapper.hpp>

using mgcpp::internal::cuda_error_t;

TEST(cudaMalloc_wrapper, cudaMalloc_wrapper_success)
{
    using mgcpp::internal::cuda_mem_get_info;
    using mgcpp::internal::cuda_malloc;

    size_t free_memory_before = 0;
    cuda_mem_get_info(&free_memory_before, nullptr);

    float* ptr = nullptr;
    cuda_error_t result = cuda_malloc((void**)&ptr,
                                      sizeof(float) * 10);

    size_t free_memory_after = 0;
    cuda_mem_get_info(&free_memory_after, nullptr);

    EXPECT_GT(free_memory_before, free_memory_after);

    EXPECT_EQ( result, cuda_error_t::success );
    mgcpp::internal::cuda_free(ptr);
}

TEST(cudaMalloc_wrapper, cudaMalloc_wrapper_failure)
{
    using mgcpp::internal::cuda_mem_get_info;
    using mgcpp::internal::cuda_malloc;

    size_t free_memory = 0;
    cuda_mem_get_info(&free_memory, nullptr);

    float* ptr = nullptr;
    cuda_error_t result = cuda_malloc((void**)&ptr, free_memory * 2);

    EXPECT_NE( result,  cuda_error_t::success );
}

TEST(cudaFree_wrapper, cudaFree_wrapper_success)
{
    using mgcpp::internal::cuda_mem_get_info;
    using mgcpp::internal::cuda_malloc;

    size_t free_memory_before = 0;
    cuda_mem_get_info(&free_memory_before, nullptr);

    float* ptr = nullptr;
    cuda_error_t malloc_result =
        mgcpp::internal::cuda_malloc((void**)&ptr,
                                     sizeof(float) * 10);
    EXPECT_EQ(malloc_result, cuda_error_t::success );

    size_t free_memory_after_malloc = 0;
    cuda_mem_get_info(&free_memory_after_malloc, nullptr);

    EXPECT_GT(free_memory_before, free_memory_after_malloc);

    cuda_error_t free_result = 
        mgcpp::internal::cuda_free((void*)ptr);
    EXPECT_EQ( free_result, cuda_error_t::success );

    size_t free_memory_after_free = 0;
    cuda_mem_get_info(&free_memory_after_free, nullptr);

    EXPECT_EQ(free_memory_after_free, free_memory_before);
}
