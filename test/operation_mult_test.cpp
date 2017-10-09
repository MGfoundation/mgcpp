
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <vector>
#include <cassert>
#include <iostream>

#include <gtest/gtest.h>

#include <mgcpp/gpu_matrix.hpp>
#include <mgcpp/operations/mult.hpp>

TEST(operation_mult, row_major_multiplication)
{
    mgcpp::cpu::matrix<float> A_init_mat(2, 4, 2);
    mgcpp::cpu::matrix<float> B_init_mat(4, 3, 4);

    mgcpp::gpu::matrix<float> A_mat(A_init_mat);
    mgcpp::gpu::matrix<float> B_mat(B_init_mat);

    auto C_mat = mgcpp::mult(A_mat, B_mat);

    auto C_mat_host = C_mat.copy_to_host();
    
    auto shape = C_mat_host.shape();
    EXPECT_EQ(shape.first, 2);
    EXPECT_EQ(shape.second, 3);

    // for(size_t i = 0; i < shape.first; ++i)
    // {
    //     for(size_t j = 0; j < shape.second; ++j)
    //     {
    //         EXPECT_EQ(C_mat_host(i, j), 32)
    //             << "i: " << i << " j: " << j; 
    //     } 
    // }
}
