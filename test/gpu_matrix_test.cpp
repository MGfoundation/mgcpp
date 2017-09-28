
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#define private public

#include <mgcpp/gpu_matrix.hpp>

TEST(gpu_matrix, default_constructor)
{
    mgcpp::gpu::matrix<float, 0> mat;

    EXPECT_EQ(mat.rows(), 0u);
    EXPECT_EQ(mat.columns(), 0u);
    EXPECT_EQ(mat._data, nullptr);
    EXPECT_EQ(mat._context, nullptr);
    EXPECT_TRUE(mat._released);
}
