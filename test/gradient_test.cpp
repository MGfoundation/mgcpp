#include <gtest/gtest.h>
#include <mgcpp/expressions/gradients.hpp>
#include <mgcpp/expressions/inspect_graph.hpp>
#include <mgcpp/mgcpp.hpp>

#include <iostream>

TEST(gradient_test, dmat_placeholder) {
  mgcpp::placeholder_node<0, mgcpp::device_matrix<float>> ph0;
  auto sum = mgcpp::reduce_sum(ph0);
  auto grad = mgcpp::grad(sum, ph0);
  mgcpp::eval_context ctx;
  mgcpp::device_matrix<float> mat({2, 4}, 3);
  ctx.feed(ph0, mat);
  auto result = grad.eval(ctx);
  EXPECT_EQ(result.shape(), mgcpp::make_shape(2, 4));
  for (size_t i = 0; i < result.shape()[0]; ++i) {
    for (size_t j = 0; j < result.shape()[1]; ++j) {
      EXPECT_FLOAT_EQ(result.check_value(i, j), 1);
    }
  }
}

TEST(gradient_test, dmat_placeholder_disconnected) {
  mgcpp::placeholder_node<0, mgcpp::device_matrix<float>> ph0;
  mgcpp::placeholder_node<1, mgcpp::device_matrix<float>> ph1;
  auto grad = mgcpp::grad(reduce_sum(ph0), ph1);
  mgcpp::eval_context ctx;
  mgcpp::device_matrix<float> mat({2, 4}, 3);
  ctx.feed(ph0, mat);
  ctx.feed(ph1, mat);
  auto result = grad.eval(ctx);
  EXPECT_EQ(result.shape(), mgcpp::make_shape(2, 4));
  for (size_t i = 0; i < result.shape()[0]; ++i) {
    for (size_t j = 0; j < result.shape()[1]; ++j) {
      EXPECT_FLOAT_EQ(result.check_value(i, j), 0);
    }
  }
}

TEST(gradient_test, dmat_add) {
  mgcpp::placeholder_node<0, mgcpp::device_matrix<float>> ph0;
  mgcpp::placeholder_node<1, mgcpp::device_matrix<float>> ph1;
  auto add = ph0 + ph1;
  std::cout << add << std::endl;

  auto grad = mgcpp::grad(reduce_sum(add), ph0);
  std::cout << grad << std::endl;

  mgcpp::eval_context ctx;
  mgcpp::device_matrix<float> mat({2, 4}, 3);
  ctx.feed(ph0, mat);
  ctx.feed(ph1, mat);
  auto result = grad.eval(ctx);
  EXPECT_EQ(result.shape(), mgcpp::make_shape(2, 4));
  for (size_t i = 0; i < result.shape()[0]; ++i) {
    for (size_t j = 0; j < result.shape()[1]; ++j) {
      EXPECT_FLOAT_EQ(result.check_value(i, j), 1);
    }
  }
}

template <typename Expr, size_t PlaceholderID, typename ResultType>
auto numerical_differentiation(
    mgcpp::scalar_expr<Expr> const& expr,
    mgcpp::placeholder_node<PlaceholderID, ResultType> wrt,
    mgcpp::eval_context ctx) {
  double R = (~expr).eval(ctx);

  ResultType x = ctx.get_placeholder<PlaceholderID, ResultType>();
  ResultType result = x;
  double epsilon = 0.00001;
  for (size_t i = 0; i < x.shape()[0]; ++i) {
    for (size_t j = 0; j < x.shape()[1]; ++j) {
      auto orig_val = x.check_value(i, j);
      x.set_value(i, j, orig_val + epsilon);
      ctx.feed(wrt, x);
      double Rp = (~expr).eval(ctx);
      double p = (Rp - R) / epsilon;
      result.set_value(i, j, p);
      x.set_value(i, j, orig_val);
    }
  }

  return result;
}

TEST(gradient_test, dmat_mul) {
  mgcpp::placeholder_node<0, mgcpp::device_matrix<double>> x;
  mgcpp::placeholder_node<1, mgcpp::device_matrix<double>> w;
  auto expr = x * w;
  auto sum = reduce_sum(expr);
  std::cout << sum << std::endl;

  auto grad = mgcpp::grad(sum, x);
  std::cout << grad << std::endl;

  mgcpp::eval_context ctx;
  auto mat1 = mgcpp::device_matrix<double>::from_list({{1, 2, 3, 4},
                                                       {2, 3, -4, 5},
                                                       {6, -7, 8, 9},
                                                       {10, 2, 3, 4},
                                                       {1, 0, 1, -2}});  // 5x4
  auto mat2 = mgcpp::device_matrix<double>::from_list(
      {{1, 2, 3}, {2, 3, 4}, {5, 6, 1}, {-1, 2, 3}});  // 4x3
  ctx.feed(x, mat1);
  ctx.feed(w, mat2);
  auto result = grad.eval(ctx);

  auto numeric_approx = numerical_differentiation(sum, x, ctx);

  EXPECT_EQ(result.shape(), mgcpp::make_shape(5, 4));
  for (size_t i = 0; i < result.shape()[0]; ++i) {
    for (size_t j = 0; j < result.shape()[1]; ++j) {
      EXPECT_FLOAT_EQ(result.check_value(i, j),
                      numeric_approx.check_value(i, j));
    }
  }
}
/*
TEST(gradient_test, dmat_dvec_mul_add) {
  mgcpp::placeholder_node<0, mgcpp::device_matrix<double>> W;
  mgcpp::placeholder_node<1, mgcpp::device_vector<double>> v;
  mgcpp::placeholder_node<1, mgcpp::device_vector<double>> b;
  auto expr = W * v + b;
  auto sum = reduce_sum(expr);
  std::cout << sum << std::endl;

  auto grad = mgcpp::grad(sum, W);
  std::cout << grad << std::endl;

  mgcpp::eval_context ctx;
  ctx.feed(W, mgcpp::device_matrix<double>::from_list({{1, 2, 3, 4},
                                                       {2, 3, -4, 5},
                                                       {6, -7, 8, 9},
                                                       {10, 2, 3, 4},
                                                       {1, 0, 1, -2}}));  // 5x4
  ctx.feed(v, mgcpp::device_vector<double>({-3, 4, -5, 6})); // 4x1
  ctx.feed(b, mgcpp::device_vector<double>({1, 2, 3, 4, 5})); // 5x1
  auto result = grad.eval(ctx);

  auto numeric_approx = numerical_differentiation(sum, W, ctx);

  EXPECT_EQ(result.shape(), mgcpp::make_shape(5, 4));
  for (size_t i = 0; i < result.shape()[0]; ++i) {
    for (size_t j = 0; j < result.shape()[1]; ++j) {
      EXPECT_FLOAT_EQ(result.check_value(i, j),
                      numeric_approx.check_value(i, j));
    }
  }
}
*/
