
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/expressions/forward.hpp>
#include <mgcpp/matrix/device_matrix.hpp>

#include <cmath>

TEST(add_expr, row_major_mat_mat_add) {
  using matrix = mgcpp::device_matrix<float>;

  matrix A_mat(mgcpp::make_shape(4, 3), 2);
  matrix B_mat(mgcpp::make_shape(4, 3), 4);

  auto add_expr = mgcpp::ref(A_mat) + mgcpp::ref(B_mat);

  matrix C_mat;
  EXPECT_NO_THROW({ C_mat = eval(add_expr); });

  auto shape = C_mat.shape();
  EXPECT_EQ(C_mat.shape(), A_mat.shape());

  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      EXPECT_EQ(C_mat.check_value(i, j), 6) << "i: " << i << " j: " << j;
    }
  }
}

TEST(add_expr, row_major_mat_mat_add_func) {
  using matrix = mgcpp::device_matrix<float>;

  matrix A_mat(mgcpp::make_shape(4, 3), 2);
  matrix B_mat(mgcpp::make_shape(4, 3), 4);

  auto add_expr = mgcpp::add(mgcpp::ref(A_mat), mgcpp::ref(B_mat));

  matrix C_mat;
  EXPECT_NO_THROW({ C_mat = eval(add_expr); });

  auto shape = C_mat.shape();
  EXPECT_EQ(C_mat.shape(), A_mat.shape());

  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      EXPECT_EQ(C_mat.check_value(i, j), 6) << "i: " << i << " j: " << j;
    }
  }
}

TEST(elemwise_expr, abs_expr) {
  using vector = mgcpp::device_vector<float>;

  vector v{1, -2, 3, -4, 5};
  auto abs_expr = mgcpp::abs(mgcpp::ref(v));

  vector result;
  EXPECT_NO_THROW({ result = eval(abs_expr); });

  EXPECT_EQ(result.shape(), 5);

  float expected[] = {1, 2, 3, 4, 5};
  for (size_t i = 0; i < result.shape(); ++i)
    EXPECT_FLOAT_EQ(result.check_value(i), expected[i]) << "i : " << i;
}

TEST(elemwise_expr, sin_expr) {
  using vector = mgcpp::device_vector<float>;

  vector v{1, -2, 3, -4, 5};
  auto sin_expr = mgcpp::sin(mgcpp::ref(v));

  vector result;
  EXPECT_NO_THROW({ result = eval(sin_expr); });

  EXPECT_EQ(result.shape(), 5);

  float expected[] = {float(std::sin(1)), float(std::sin(-2)),
                      float(std::sin(3)), float(std::sin(-4)),
                      float(std::sin(5))};
  for (size_t i = 0; i < result.shape(); ++i)
    EXPECT_FLOAT_EQ(result.check_value(i), expected[i]) << "i : " << i;
}

TEST(caching, caching) {
    mgcpp::device_matrix<float> a(mgcpp::make_shape(3, 3)),
                                b(mgcpp::make_shape(3, 3));
    auto c = ref(a) + ref(b);
    auto d = trans(c);
    auto r = c * d;

    mgcpp::eval_context ctx;
    auto result = mgcpp::eval(r, ctx);
    EXPECT_EQ(result.shape(), mgcpp::make_shape(3, 3));
    EXPECT_EQ(ctx.cache_hits, 1);
}
