
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda/internal/status_wrapper.hpp>
#include <mgcpp/cuda/internal/cuda_error.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

TEST(cuda_status, cuda_mem_get_info)
{
    using mgcpp::internal::cuda_error_t;
    using mgcpp::internal::cuda_mem_get_info;

    size_t free = 0;
    size_t total = 0;

    cudaError_t result = cudaMemGetInfo(&free, &total);

    EXPECT_EQ(result, cudaError_t::cudaSuccess);

    size_t free_wrapper = 0;
    size_t total_wrapper = 0;

    cuda_error_t result_wrapper =
        cuda_mem_get_info(&free_wrapper, &total_wrapper);

    EXPECT_EQ(static_cast<cuda_error_t>(result_wrapper),
              cuda_error_t::success);

    EXPECT_EQ(free_wrapper, free);
    EXPECT_EQ(total_wrapper, total);
}
