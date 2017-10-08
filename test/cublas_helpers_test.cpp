
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/cublas/cublas_helpers.hpp>
#include <mgcpp/cuda/memory.hpp>

#include <algorithm>
#include <cstdlib>

TEST(cublas_matrix_memcpy, set_and_get)
{
    size_t row = 4;
    size_t col = 2;

    float* ptr = (float*)malloc(sizeof(float) * row * col);
    auto device_ptr = mgcpp::cuda_malloc<float>(row * col);

    EXPECT_TRUE(device_ptr);

    for(auto i = 0u; i < row * col; ++i)
    {
        ptr[i] = i;
    }

    auto set_result =
        mgcpp::cublas_set_matrix(row, col,
                                 ptr, device_ptr.value());
    EXPECT_TRUE(set_result);

    float* destination = (float*)malloc(sizeof(float) * row * col);

    auto get_result =
        mgcpp::cublas_get_matrix(row, col,
                                 device_ptr.value(), destination);
    EXPECT_TRUE(get_result);

    EXPECT_TRUE(std::equal(ptr, ptr + row * col, destination));

    (void)mgcpp::cuda_free(device_ptr.value());
    free(destination);
    free(ptr);
}
