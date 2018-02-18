
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/expressions/forward.hpp>
#include <mgcpp/matrix/device_matrix.hpp>

TEST(mult_expr, dmat_dmat_mult) {
  {
    using matrix = mgcpp::device_matrix<float>;

    matrix A_mat({2, 4}, 2);
    matrix B_mat({4, 3}, 4);

    auto mult_expr = A_mat * B_mat;

    matrix C_mat;
    EXPECT_NO_THROW({ C_mat = mult_expr.eval(); });

    auto shape = C_mat.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);

    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        EXPECT_EQ(C_mat.check_value(i, j), 32) << "i: " << i << " j: " << j;
      }
    }
  }
  {
    using matrix = mgcpp::device_matrix<float>;

    matrix A_mat({2, 4}, 2);
    matrix B_mat({4, 3}, 4);

    auto mult_expr = mgcpp::mult(A_mat, B_mat);

    matrix C_mat;
    EXPECT_NO_THROW({ C_mat = mult_expr.eval(); });

    auto shape = C_mat.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);

    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        EXPECT_EQ(C_mat.check_value(i, j), 32) << "i: " << i << " j: " << j;
      }
    }
  }
  {
    using matrix = mgcpp::device_matrix<float>;

    matrix A_mat({2, 4}, 2);
    matrix B_mat({4, 3}, 4);

    auto mult_expr = 1.0 * A_mat * B_mat;

    matrix C_mat;
    EXPECT_NO_THROW({ C_mat = mult_expr.eval(); });

    auto shape = C_mat.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);

    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        EXPECT_EQ(C_mat.check_value(i, j), 32) << "i: " << i << " j: " << j;
      }
    }
  }
  {
    using matrix = mgcpp::device_matrix<float>;

    matrix A_mat({2, 4}, 2);
    matrix B_mat({4, 3}, 4);

    auto mult_expr = A_mat * B_mat * 1.0;

    matrix C_mat;
    EXPECT_NO_THROW({ C_mat = mult_expr.eval(); });

    auto shape = C_mat.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);

    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        EXPECT_EQ(C_mat.check_value(i, j), 32) << "i: " << i << " j: " << j;
      }
    }
  }
}

TEST(mult_expr, dmat_dmat_add) {
  using matrix = mgcpp::device_matrix<float>;

  matrix A_mat({2, 4}, 2);
  matrix B_mat({4, 3}, 4);
  matrix C_mat({2, 3}, 3);

  auto add_expr = (A_mat * B_mat) + C_mat;

  matrix D_mat;
  EXPECT_NO_THROW({ D_mat = add_expr.eval(); });

  auto shape = D_mat.shape();
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);

  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      EXPECT_EQ(D_mat.check_value(i, j), 35) << "i: " << i << " j: " << j;
    }
  }
}

TEST(mult_expr, scalar_dmat_mult) {
  {
    using matrix = mgcpp::device_matrix<double>;

    matrix A_mat({2, 4}, 2);

    auto expr = 7.0 * A_mat;

    matrix B_mat;
    EXPECT_NO_THROW({ B_mat = expr.eval(); });

    auto shape = B_mat.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 4);

    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        EXPECT_EQ(B_mat.check_value(i, j), 14) << "i: " << i << " j: " << j;
      }
    }
  }

  // {
  //     using matrix = mgcpp::device_matrix<mgcpp::half>;

  //     matrix A_mat({2, 4}, mgcpp::half(2.0));

  //     auto expr =  A_mat * 7.0;

  //     matrix B_mat;
  //     EXPECT_NO_THROW({B_mat = expr.eval();});

  //     auto shape = B_mat.shape();
  //     EXPECT_EQ(shape[0], 2);
  //     EXPECT_EQ(shape[1], 4);

  //     for(size_t i = 0; i < shape[0]; ++i)
  //     {
  //         for(size_t j = 0; j < shape[1]; ++j)
  //         {
  //             EXPECT_EQ(half_float::half_cast<int>(B_mat.check_value(i, j)),
  //             14)
  //                 << "i: " << i << " j: " << j;
  //         }
  //     }
  // }

  {
    using matrix = mgcpp::device_matrix<mgcpp::complex<float>>;

    matrix A_mat({2, 4}, std::complex<float>{1, 2});

    auto expr = mgcpp::mult(7.0, A_mat);

    matrix B_mat;
    EXPECT_NO_THROW({ B_mat = expr.eval(); });

    auto shape = B_mat.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 4);

    auto answer = std::complex<float>(7, 14);
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        EXPECT_EQ(B_mat.check_value(i, j), answer) << "i: " << i << " j: " << j;
      }
    }
  }

  {
    using matrix = mgcpp::device_matrix<mgcpp::complex<double>>;

    matrix A_mat({2, 4}, std::complex<double>{1, 2});

    auto expr = mgcpp::mult(A_mat, 7.0);

    matrix B_mat;
    EXPECT_NO_THROW({ B_mat = expr.eval(); });

    auto shape = B_mat.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 4);

    auto answer = std::complex<double>(7, 14);
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        EXPECT_EQ(B_mat.check_value(i, j), answer) << "i: " << i << " j: " << j;
      }
    }
  }
}

TEST(mat_expression, mat_trans) {
  auto mat =
      mgcpp::device_matrix<float>::from_list({{-1, -2, -3}, {-4, -5, -6}});

  auto expr = mgcpp::trans(mat);
  mgcpp::device_matrix<float> result{};
  EXPECT_NO_THROW({ result = expr.eval(); });
  EXPECT_EQ(result.shape()[0], 3);
  EXPECT_EQ(result.shape()[1], 2);
  EXPECT_EQ(result.check_value(0, 0), -1);
  EXPECT_EQ(result.check_value(0, 1), -4);
  EXPECT_EQ(result.check_value(1, 0), -2);
  EXPECT_EQ(result.check_value(1, 1), -5);
  EXPECT_EQ(result.check_value(2, 0), -3);
  EXPECT_EQ(result.check_value(2, 1), -6);
}

TEST(mat_expression, mat_trans_gemm_add) {
  auto mat =
      mgcpp::device_matrix<float>::from_list({{-1, -2, -3}, {-4, -5, -6}});

  auto mat2 =
      mgcpp::device_matrix<float>::from_list({{-1, -2, -3}, {-4, -5, -6}});

  auto mat3 = mgcpp::device_matrix<float>::from_list(
      {{-1, -2, -3}, {-4, -5, -6}, {-7, -8, -9}});

  auto expr = mgcpp::trans(mat) * mat2 + mat3;

  mgcpp::device_matrix<float> result{};
  EXPECT_NO_THROW({ result = expr.eval(); });

  EXPECT_EQ(result.shape()[0], 3);
  EXPECT_EQ(result.shape()[1], 3);

  EXPECT_EQ(result.check_value(0, 0), 16);
  EXPECT_EQ(result.check_value(0, 1), 20);
  EXPECT_EQ(result.check_value(0, 2), 24);
  EXPECT_EQ(result.check_value(1, 0), 18);
  EXPECT_EQ(result.check_value(1, 1), 24);
  EXPECT_EQ(result.check_value(1, 2), 30);
  EXPECT_EQ(result.check_value(2, 0), 20);
  EXPECT_EQ(result.check_value(2, 1), 28);
  EXPECT_EQ(result.check_value(2, 2), 36);
}
