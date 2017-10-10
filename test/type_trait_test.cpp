
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/type_traits/mat_mat_expr.hpp>

#include <gtest/gtest.h>

TEST(mat_mat_expr_trait, is_same_gpu_matrix_success)
{
    using FirstMat = mgcpp::gpu::matrix<float>; 
    using SecondMat = mgcpp::gpu::matrix<float>; 

    EXPECT_TRUE(is_same_gpu_matrix<FirstMat, SecondMat>::value);
}

TEST(mat_mat_expr_trait, is_same_gpu_matrix_fail)
{
    using FirstMat = mgcpp::gpu::matrix<float>; 
    using SecondMat = mgcpp::gpu::matrix<double>; 

    EXPECT_FALSE(is_same_gpu_matrix<FirstMat, SecondMat>::value);
}
