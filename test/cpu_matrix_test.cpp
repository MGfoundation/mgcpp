
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#define private public
#include <mgcpp/cpu_matrix.hpp>

TEST(cpu_matrix, default_constructor)
{
    mgcpp::cpu::matrix<float> mat{};

    EXPECT_EQ(mat.rows(), 0);
    EXPECT_EQ(mat.columns(), 0);
    EXPECT_EQ(mat._data, nullptr);
}

TEST(cpu_matrix, allocating_constructor)
{
    size_t row_dim = 5;
    size_t col_dim = 10;
    mgcpp::cpu::matrix<float> mat{row_dim, col_dim};

    EXPECT_NE(mat._data, nullptr);
    EXPECT_EQ(row_dim, mat.rows());
    EXPECT_EQ(col_dim, mat.columns());
}

TEST(cpu_matrix, allocating_initializing_constructor)
{
    size_t row_dim = 5;
    size_t col_dim = 10;
    float init_val = 7;
    mgcpp::cpu::matrix<float> mat{row_dim, col_dim, init_val};

    EXPECT_NE(mat._data, nullptr);
    EXPECT_EQ(row_dim, mat.rows());
    EXPECT_EQ(col_dim, mat.columns());

    EXPECT_NO_THROW(
        {
            for(size_t i = 0; i < row_dim; ++i)
            {
                for(size_t j = 0; j < col_dim; ++j) 
                {
                    EXPECT_EQ(mat(i, j), init_val);
                }
            }
        });
}

TEST(cpu_matrix, non_const_parenthese_operator)
{
    size_t row_dim = 5;
    size_t col_dim = 10;
    mgcpp::cpu::matrix<float> mat{row_dim, col_dim};

    float counter = 0;
    for(size_t i = 0; i < row_dim; ++i)
    {
        for(size_t j = 0; j < row_dim; ++j)
        {
            mat(i, j) = counter;
            ++counter;
        }
    }

    counter = 0;
    for(size_t i = 0; i < row_dim; ++i)
    {
        for(size_t j = 0; j < row_dim; ++j)
        {
            EXPECT_NO_THROW({EXPECT_EQ(mat(i, j), counter);});
            ++counter;
        }
    }
}

