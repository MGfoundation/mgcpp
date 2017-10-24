
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/gpu_matrix.hpp>
#include <mgcpp/expressions/mat_mat_mult_expr.hpp>

TEST(mult_expr, row_major_mat_mat_mult)
{
    using matrix = mgcpp::gpu::matrix<float>;

    matrix A_mat(2, 4, 2);
    matrix B_mat(4, 3, 4);

    auto mult_expr = A_mat * B_mat;

    matrix C_mat; 
    EXPECT_NO_THROW({C_mat = mult_expr.eval();});
    
    auto shape = C_mat.shape();
    EXPECT_EQ(shape.first, 2);
    EXPECT_EQ(shape.second, 3);

    for(size_t i = 0; i < shape.first; ++i)
    {
        for(size_t j = 0; j < shape.second; ++j)
        {
            EXPECT_EQ(C_mat.check_value(i, j), 32)
                << "i: " << i << " j: " << j; 
        } 
    }
}
