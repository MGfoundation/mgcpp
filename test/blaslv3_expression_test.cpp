
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/expressions/dmat_dmat_mult.hpp>

TEST(mult_expr, row_major_mat_mat_mult)
{
    using matrix = mgcpp::device_matrix<float>;

    matrix A_mat({2, 4}, 2);
    matrix B_mat({4, 3}, 4);

    auto mult_expr = A_mat * B_mat;

    matrix C_mat; 
    EXPECT_NO_THROW({C_mat = mult_expr.eval();});
    
    auto shape = C_mat.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);

    for(size_t i = 0; i < shape[0]; ++i)
    {
        for(size_t j = 0; j < shape[1]; ++j)
        {
            EXPECT_EQ(C_mat.check_value(i, j), 32)
                << "i: " << i << " j: " << j; 
        } 
    }
}


TEST(mult_expr, row_major_mat_mat_mult_func)
{
    using matrix = mgcpp::device_matrix<float>;

    matrix A_mat({2, 4}, 2);
    matrix B_mat({4, 3}, 4);

    auto mult_expr = mgcpp::mult(A_mat, B_mat);

    matrix C_mat; 
    EXPECT_NO_THROW({C_mat = mult_expr.eval();});
    
    auto shape = C_mat.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);

    for(size_t i = 0; i < shape[0]; ++i)
    {
        for(size_t j = 0; j < shape[1]; ++j)
        {
            EXPECT_EQ(C_mat.check_value(i, j), 32)
                << "i: " << i << " j: " << j; 
        } 
    }
}
