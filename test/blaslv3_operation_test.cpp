
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <vector>
#include <cassert>

#include <gtest/gtest.h>

#include <mgcpp/gpu_matrix.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/operations/add.hpp>

TEST(mat_mat_operation, row_major_multiplication)
{
    mgcpp::gpu::matrix<float> A_mat(2, 4, 2);
    mgcpp::gpu::matrix<float> B_mat(4, 3, 4);

    auto C_mat = mgcpp::strict::mult(A_mat, B_mat);

    auto C_mat_host = C_mat.copy_to_host();
    
    auto shape = C_mat_host.shape();
    EXPECT_EQ(shape.first, 2);
    EXPECT_EQ(shape.second, 3);

    for(size_t i = 0; i < shape.first; ++i)
    {
        for(size_t j = 0; j < shape.second; ++j)
        {
            EXPECT_EQ(C_mat_host(i, j), 32)
                << "i: " << i << " j: " << j; 
        } 
    }
}

TEST(mat_mat_operation , row_major_addition)
{
    mgcpp::gpu::matrix<float> A_mat(4, 2, 2);
    mgcpp::gpu::matrix<float> B_mat(4, 2, 4);

    auto C_mat = mgcpp::strict::add(A_mat, B_mat);

    auto C_mat_host = C_mat.copy_to_host();
    
    auto shape = C_mat_host.shape();
    EXPECT_EQ(shape.first, 4);
    EXPECT_EQ(shape.second, 2);

    for(size_t i = 0; i < shape.first; ++i)
    {
        for(size_t j = 0; j < shape.second; ++j)
        {
            EXPECT_EQ(C_mat_host(i, j), 6)
                << "i: " << i << " j: " << j; 
        } 
    }
}
