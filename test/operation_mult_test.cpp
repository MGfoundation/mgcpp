
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <cassert>

#include <gtest/gtest.h>

#include <mgcpp/gpu_matrix.hpp>
#include <mgcpp/operations/mult.hpp>

TEST(operation_mult, row_major_multiplication)
{
    mgcpp::cpu::matrix<float> A_init_mat(2, 4, 2);
    mgcpp::cpu::matrix<float> B_init_mat(4, 2, 4);

    mgcpp::gpu::matrix<float> A_mat(2, 4);
    A_mat.copy_from_host(A_init_mat);

    mgcpp::gpu::matrix<float> B_mat(4, 2);
    B_mat.copy_from_host(B_init_mat);

    auto C_mat = mgcpp::mult(A_mat, B_mat);

    auto shape = C_mat.shape();
    EXPECT_EQ(shape.first, 2);
    EXPECT_EQ(shape.second, 2);

    printf("safe?\n");
    for(size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 2; ++j)
        {
            EXPECT_EQ(C_mat.check_value(i, j), 32); 
        } 
    }
}
