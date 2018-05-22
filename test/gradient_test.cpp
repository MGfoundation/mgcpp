#include <gtest/gtest.h>
#include <mgcpp/mgcpp.hpp>
#include <mgcpp/expressions/gradients.hpp>
#include <mgcpp/expressions/inspect_graph.hpp>

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

TEST(gradient_test, dmat_mul) {
  mgcpp::placeholder_node<0, mgcpp::device_matrix<double>> ph0;
  mgcpp::placeholder_node<1, mgcpp::device_matrix<double>> ph1;
  auto mul = ph0 * ph1;
  auto sum = reduce_sum(mul);
  auto grad = mgcpp::grad(sum, ph0);

  mgcpp::eval_context ctx;
  mgcpp::device_matrix<double> mat1({4, 4}, 3);
  mgcpp::device_matrix<double> mat2({4, 4}, 2);
  ctx.feed(ph0, mat1);
  ctx.feed(ph1, mat2);
  auto result = grad.eval(ctx);

  mgcpp::eval_context ctx2;
  double epsilon = 0.00001;
  mat1.set_value(0, 0, 3 + epsilon);
  ctx2.feed(ph0, mat1);
  ctx2.feed(ph1, mat2);
  double R = sum.eval(ctx);
  double Rp = sum.eval(ctx2);
  auto numeric_approx = (Rp - R) / epsilon;
  EXPECT_EQ(result.shape(), mgcpp::make_shape(4, 4));
  for (size_t i = 0; i < result.shape()[0]; ++i) {
      for (size_t j = 0; j < result.shape()[1]; ++j) {
          EXPECT_FLOAT_EQ(result.check_value(i, j), numeric_approx);
      }
  }
}

TEST(gradient_test, dmat_mul_add) {
  mgcpp::placeholder_node<0, mgcpp::device_matrix<double>> x;
  mgcpp::placeholder_node<1, mgcpp::device_matrix<double>> w;
  auto expr = x * w + x * x;
  auto sum = reduce_sum(expr);
  std::cout << sum << std::endl;

  auto grad = mgcpp::grad(sum, x);
  std::cout << grad << std::endl;

  mgcpp::eval_context ctx;
  mgcpp::device_matrix<double> mat1({4, 4}, 3);
  mgcpp::device_matrix<double> mat2({4, 4}, 2);
  ctx.feed(x, mat1);
  ctx.feed(w, mat2);
  auto result = grad.eval(ctx);

  mgcpp::eval_context ctx2;
  double epsilon = 0.00001;
  mat1.set_value(0, 0, 3 + epsilon);
  ctx2.feed(x, mat1);
  ctx2.feed(w, mat2);
  double R = sum.eval(ctx);
  double Rp = sum.eval(ctx2);
  auto numeric_approx = (Rp - R) / epsilon;
  EXPECT_EQ(result.shape(), mgcpp::make_shape(4, 4));
  for (size_t i = 0; i < result.shape()[0]; ++i) {
      for (size_t j = 0; j < result.shape()[1]; ++j) {
          EXPECT_FLOAT_EQ(result.check_value(i, j), numeric_approx);
      }
  }
}
