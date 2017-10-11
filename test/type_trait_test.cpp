
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cpu/forward.hpp>
#include <mgcpp/type_traits/mat_mat_expr.hpp>
#include <mgcpp/type_traits/gpu_mat.hpp>

#include <gtest/gtest.h>

TEST(mat_mat_expr_trait, is_same_gpu_matrix_success)
{
    using FirstMat = mgcpp::gpu::matrix<float>; 
    using SecondMat = mgcpp::gpu::matrix<float>; 
    bool is_same_gpu_matrix =
        mgcpp::is_same_gpu_matrix<FirstMat, SecondMat>::value;

    EXPECT_TRUE(is_same_gpu_matrix);
}

TEST(mat_mat_expr_trait, is_same_gpu_matrix_fail)
{
    using FirstMat = mgcpp::gpu::matrix<float>; 
    using SecondMat = mgcpp::gpu::matrix<double>; 

    bool is_same_gpu_matrix =
        mgcpp::is_same_gpu_matrix<FirstMat, SecondMat>::value;

    EXPECT_FALSE(is_same_gpu_matrix);
}

TEST(gpu_matrix_trait, is_gpu_matrix_success)
{
    using mat = mgcpp::gpu::matrix<float>;

    bool is_gpu_matrix = mgcpp::is_gpu_matrix<mat>::value;
    EXPECT_TRUE(is_gpu_matrix);
}

TEST(gpu_matrix_trait, is_gpu_matrix_cv_success)
{
    using mat = mgcpp::gpu::matrix<float>;

    bool gpu_matrix = mgcpp::is_gpu_matrix<mat>::value;
    EXPECT_TRUE(gpu_matrix);

    bool gpu_matrix_ref =
        mgcpp::is_gpu_matrix<std::decay<mat&>::type>::value;
    EXPECT_TRUE(gpu_matrix_ref);

    bool gpu_matrix_const_ref =
        mgcpp::is_gpu_matrix<std::decay<mat const&>::type>::value;
    EXPECT_TRUE(gpu_matrix_const_ref);
}

TEST(gpu_matrix_trait, is_gpu_matrix_fail)
{
    using mat = mgcpp::cpu::matrix<float>;

    bool is_gpu_matrix = mgcpp::is_gpu_matrix<mat>::value;
    EXPECT_FALSE(is_gpu_matrix);

}
