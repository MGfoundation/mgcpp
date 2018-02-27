
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/operations/map.hpp>
#include <mgcpp/operations/add.hpp>
#include <mgcpp/operations/hdmd.hpp>
#include <mgcpp/operations/mean.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/operations/pad.hpp>
#include <mgcpp/operations/sub.hpp>
#include <mgcpp/operations/sum.hpp>
#include <mgcpp/vector/device_vector.hpp>

TEST(vec_vec_operation, vec_sum) {
  size_t size = 10;
  float init_val = 3;
  mgcpp::device_vector<float> vec(size, init_val);

  float result = 0;
  EXPECT_NO_THROW({ result = mgcpp::strict::sum(vec); });

  EXPECT_EQ(result, static_cast<float>(size) * init_val);
}

TEST(vec_vec_operation, vec_add) {
  size_t size = 5;
  float first_init_val = 3;
  mgcpp::device_vector<float> first(size, first_init_val);

  float second_init_val = 4;
  mgcpp::device_vector<float> second(size, second_init_val);

  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({ result = mgcpp::strict::add(first, second); });

  for (auto i = 0u; i < size; ++i) {
    EXPECT_EQ(result.check_value(i), first_init_val + second_init_val);
  }
}

TEST(vec_vec_operation, vec_sub) {
  size_t size = 5;
  float first_init_val = 4;
  mgcpp::device_vector<float> first(size, first_init_val);

  float second_init_val = 3;
  mgcpp::device_vector<float> second(size, second_init_val);

  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({ result = mgcpp::strict::sub(first, second); });

  for (auto i = 0u; i < size; ++i) {
    EXPECT_EQ(result.check_value(i), first_init_val - second_init_val);
  }
}

TEST(vec_operation, vec_scalar_mult) {
  size_t size = 5;
  float init_val = 3;
  mgcpp::device_vector<float> vec(size, init_val);

  float scalar = 4;

  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({ result = mgcpp::strict::mult(scalar, vec); });

  for (auto i = 0u; i < size; ++i) {
    EXPECT_EQ(result.check_value(i), init_val * scalar);
  }
}

TEST(vec_operation, vec_abs) {
  mgcpp::device_vector<float> vec{-1, 2, -3};

  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({
    result = mgcpp::strict::abs(vec);

    EXPECT_EQ(result.check_value(0), 1);
    EXPECT_EQ(result.check_value(1), 2);
    EXPECT_EQ(result.check_value(2), 3);
  });
}

TEST(vec_operation, vec_sin) {
  mgcpp::device_vector<float> vec{-1, -2, -3};

  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({
    result = mgcpp::strict::sin(vec);

    EXPECT_FLOAT_EQ(result.check_value(0), -0.841470985f);
    EXPECT_FLOAT_EQ(result.check_value(1), -0.909297427f);
    EXPECT_FLOAT_EQ(result.check_value(2), -0.141120008f);
  });
}

TEST(vec_operation, vec_cos) {
  mgcpp::device_vector<float> vec{-1, -2, -3};

  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({
    result = mgcpp::strict::cos(vec);

    EXPECT_FLOAT_EQ(result.check_value(0), 0.540302306);
    EXPECT_FLOAT_EQ(result.check_value(1), -0.416146837);
    EXPECT_FLOAT_EQ(result.check_value(2), -0.989992497);
  });
}

TEST(vec_operation, vec_tan) {
  mgcpp::device_vector<float> vec{-1, -2, -3};

  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({
    result = mgcpp::strict::tan(vec);

    EXPECT_FLOAT_EQ(result.check_value(0), -1.557407725);
    EXPECT_FLOAT_EQ(result.check_value(1), 2.185039863);
    EXPECT_FLOAT_EQ(result.check_value(2), 0.142546543);
  });
}

TEST(vec_operation, vec_sinh) {
  mgcpp::device_vector<float> vec{-1, -2, -3};

  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({
    result = mgcpp::strict::sinh(vec);

    EXPECT_FLOAT_EQ(result.check_value(0), -1.175201194);
    EXPECT_FLOAT_EQ(result.check_value(1), -3.626860408);
    EXPECT_FLOAT_EQ(result.check_value(2), -10.017874927);
  });
}

TEST(vec_operation, vec_cosh) {
  mgcpp::device_vector<float> vec{-1, -2, -3};

  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({
    result = mgcpp::strict::cosh(vec);

    EXPECT_FLOAT_EQ(result.check_value(0), 1.543080635);
    EXPECT_FLOAT_EQ(result.check_value(1), 3.762195691);
    EXPECT_FLOAT_EQ(result.check_value(2), 10.067661996);
  });
}

TEST(vec_operation, vec_tanh) {
  mgcpp::device_vector<float> vec{-1, -2, -3};

  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({
    result = mgcpp::strict::tanh(vec);

    EXPECT_FLOAT_EQ(result.check_value(0), -0.761594156);
    EXPECT_FLOAT_EQ(result.check_value(1), -0.96402758);
    EXPECT_FLOAT_EQ(result.check_value(2), -0.995054754);
  });
}

TEST(vec_operation, vec_relu) {
  mgcpp::device_vector<float> vec{-1, -2, 3};

  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({
    result = mgcpp::strict::relu(vec);

    EXPECT_FLOAT_EQ(result.check_value(0), 0);
    EXPECT_FLOAT_EQ(result.check_value(1), 0);
    EXPECT_FLOAT_EQ(result.check_value(2), 3);
  });
}

TEST(vec_operation, vec_mean) {
  mgcpp::device_vector<float> vec{1, 2, 3};

  EXPECT_NO_THROW({
    float result = mgcpp::strict::mean(vec);

    EXPECT_EQ(result, 2);
  });
}

TEST(vec_vec_operation, vec_hadamard_product) {
  size_t size = 5;
  float first_val = 3;
  float second_val = 5;
  mgcpp::device_vector<float> first(size, first_val);
  mgcpp::device_vector<float> second(size, second_val);

  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({ result = mgcpp::strict::hdmd(first, second); });

  for (auto i = 0u; i < size; ++i) {
    EXPECT_EQ(result.check_value(i), first_val * second_val);
  }
}

TEST(vec_operation, vec_pad) {
  size_t size = 5;
  float init_val = 3;
  mgcpp::device_vector<float> vec(size, init_val);

  size_t left_pad = 10;
  size_t right_pad = 30;
  float pad_val = -1;
  mgcpp::device_vector<float> result{};
  EXPECT_NO_THROW({
    result = mgcpp::strict::pad(vec, {left_pad, right_pad}, pad_val);
  });

  EXPECT_EQ(result.size(), left_pad + size + right_pad);
  for (auto i = 0u; i < left_pad; ++i) {
    EXPECT_EQ(result.check_value(i), pad_val);
  }
  for (auto i = left_pad; i < left_pad + size; ++i) {
    EXPECT_EQ(result.check_value(i), init_val);
  }
  for (auto i = left_pad + size; i < left_pad + size + right_pad; ++i) {
    EXPECT_EQ(result.check_value(i), pad_val);
  }
}
