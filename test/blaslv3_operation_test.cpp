
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <vector>
#include <cassert>

#include <gtest/gtest.h>

#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/operations/abs.hpp>
#include <mgcpp/operations/add.hpp>
#include <mgcpp/operations/mean.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/operations/sub.hpp>
#include <mgcpp/operations/sum.hpp>

TEST(mat_mat_operation, row_major_multiplication)
{
    mgcpp::device_matrix<float> A_mat({2, 4}, 2);
    mgcpp::device_matrix<float> B_mat({4, 3}, 4);

    auto C_mat = mgcpp::strict::mult(A_mat, B_mat);

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

TEST(mat_mat_operation , row_major_addition)
{
    mgcpp::device_matrix<float> A_mat({4, 2}, 2);
    mgcpp::device_matrix<float> B_mat({4, 2}, 4);

    auto C_mat = mgcpp::strict::add(A_mat, B_mat);

    auto shape = C_mat.shape();
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 2);

    for(size_t i = 0; i < shape[0]; ++i)
    {
        for(size_t j = 0; j < shape[1]; ++j)
        {
            EXPECT_EQ(C_mat.check_value(i, j), 6)
                << "i: " << i << " j: " << j; 
        } 
    }
}

TEST(mat_mat_operation , matrix_substraction)
{
    mgcpp::device_matrix<float> A_mat({4, 2}, 4);
    mgcpp::device_matrix<float> B_mat({4, 2}, 2);

    auto C_mat = mgcpp::strict::sub(A_mat, B_mat);

    auto shape = C_mat.shape();
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 2);

    for(size_t i = 0; i < shape[0]; ++i)
    {
        for(size_t j = 0; j < shape[1]; ++j)
        {
            EXPECT_EQ(C_mat.check_value(i, j), 2)
                << "i: " << i << " j: " << j; 
        } 
    }
}

TEST(mat_operation, mat_abs)
{
         auto mat = mgcpp::device_matrix<float>::from_list({{-1, -2, -3}, {-4, -5, -6}});

         mgcpp::device_matrix<float> result{};
         EXPECT_NO_THROW({
                 result = mgcpp::strict::abs(mat);

                 EXPECT_EQ(result.check_value(0, 0), 1);
                 EXPECT_EQ(result.check_value(0, 1), 2);
                 EXPECT_EQ(result.check_value(0, 2), 3);
                 EXPECT_EQ(result.check_value(1, 0), 4);
                 EXPECT_EQ(result.check_value(1, 1), 5);
                 EXPECT_EQ(result.check_value(1, 2), 6);
             });
}

TEST(mat_operation, mat_sum)
{
    size_t m = 5;
    size_t n = 5;
    size_t value = 5;
    mgcpp::device_matrix<float> mat({m, n}, value);

    EXPECT_NO_THROW({
            float result = mgcpp::strict::sum(mat);

            EXPECT_EQ(result, m * n * value);
        });
}

TEST(mat_operation, mat_mean)
{
    auto mat = mgcpp::device_matrix<float>::from_list({{1, 2, 3}, {1, 2, 3}});

    EXPECT_NO_THROW({
            float result = mgcpp::strict::mean(mat);

            EXPECT_EQ(result, 2);
        });
}
