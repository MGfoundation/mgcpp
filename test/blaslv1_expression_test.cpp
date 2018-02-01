
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/expressions/dmat_dmat_add.hpp>

TEST(mult_expr, row_major_mat_mat_add)
{
    using matrix = mgcpp::device_matrix<float>;

    matrix A_mat(mgcpp::make_shape(4, 3), 2);
    matrix B_mat(mgcpp::make_shape(4, 3), 4);

    auto mult_expr = A_mat + B_mat;

    matrix C_mat; 
    EXPECT_NO_THROW({C_mat = mult_expr.eval();});
    
    auto shape = C_mat.shape();
    EXPECT_EQ(C_mat.shape(), A_mat.shape());

    for(size_t i = 0; i < shape[0]; ++i)
    {
        for(size_t j = 0; j < shape[1]; ++j)
        {
            EXPECT_EQ(C_mat.check_value(i, j), 6)
                << "i: " << i << " j: " << j; 
        } 
    }
}

TEST(mult_expr, row_major_mat_mat_add_func)
{
    using matrix = mgcpp::device_matrix<float>;

    matrix A_mat(mgcpp::make_shape(4, 3), 2);
    matrix B_mat(mgcpp::make_shape(4, 3), 4);

    auto mult_expr = mgcpp::add(A_mat, B_mat);

    matrix C_mat; 
    EXPECT_NO_THROW({C_mat = mult_expr.eval();});
    
    auto shape = C_mat.shape();
    EXPECT_EQ(C_mat.shape(), A_mat.shape());

    for(size_t i = 0; i < shape[0]; ++i)
    {
        for(size_t j = 0; j < shape[1]; ++j)
        {
            EXPECT_EQ(C_mat.check_value(i, j), 6)
                << "i: " << i << " j: " << j; 
        } 
    }
}
