#include <gtest/gtest.h>
#include <mgcpp/mgcpp.hpp>
#include <mgcpp/expressions/gradients.hpp>

TEST(gradient_test, dmat_placeholder) {
  mgcpp::placeholder_node<0, mgcpp::dmat_expr, mgcpp::device_matrix<float>> ph0;
  auto grad = mgcpp::grad(ph0, ph0);
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
