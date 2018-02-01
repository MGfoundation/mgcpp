
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/kernels/mgblas_helpers.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/cuda/device.hpp>

TEST(mgblas_helpers, array_init)
{
    auto stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(stat);

    size_t size = 20;
    float value = 7;

    auto rst = mgcpp::cuda_malloc<float>(size);
    auto status = mgcpp::mgblas_fill(rst.value(), value, size);
    EXPECT_TRUE(status) << status.error();

    float* host = (float*)malloc(sizeof(float) * size);

    (void)mgcpp::cuda_memcpy(host, rst.value(), size,
                             mgcpp::cuda_memcpy_kind::device_to_host);

    for(auto i = 0u; i < size; ++i)
    {
        EXPECT_EQ(*host, value); 
    }

    free(host);
    (void)mgcpp::cuda_free(rst.value());
}
