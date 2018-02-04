
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/expressions/dmat_dmat_mult.hpp>
#include <mgcpp/expressions/scalar_dmat_mult.hpp>
#include <mgcpp/expressions/dmat_dmat_add.hpp>

TEST(mult_expr, dmat_dmat_mult)
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


TEST(mult_expr, dmat_dmat_mult_func)
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

TEST(mult_expr, dmat_dmat_mult_add)
{
    using matrix = mgcpp::device_matrix<float>;

    matrix A_mat({2, 4}, 2);
    matrix B_mat({4, 3}, 4);
    matrix C_mat({2, 3}, 3);

    auto add_expr = (A_mat * B_mat) + C_mat;

    matrix D_mat; 
    EXPECT_NO_THROW({D_mat = add_expr.eval();});
    
    auto shape = D_mat.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);

    for(size_t i = 0; i < shape[0]; ++i)
    {
        for(size_t j = 0; j < shape[1]; ++j)
        {
            EXPECT_EQ(D_mat.check_value(i, j), 35)
                << "i: " << i << " j: " << j; 
        } 
    }
}

TEST(mult_expr, scalar_dmat_mult)
{
    using matrix = mgcpp::device_matrix<float>;

    matrix A_mat({2, 4}, 2);

    auto expr = 7.0 * A_mat;

    matrix B_mat; 
    EXPECT_NO_THROW({B_mat = expr.eval();});
    
    auto shape = B_mat.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);

    for(size_t i = 0; i < shape[0]; ++i)
    {
        for(size_t j = 0; j < shape[1]; ++j)
        {
            EXPECT_EQ(B_mat.check_value(i, j), 14)
                << "i: " << i << " j: " << j; 
        } 
    }
}
