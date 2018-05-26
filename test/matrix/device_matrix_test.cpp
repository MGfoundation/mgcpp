
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <cstdlib>
#include <thread>

#include <gtest/gtest.h>

#define ERROR_CHECK_EXCEPTION true

#include <mgcpp/adapters/adapter_base.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/global/half_precision.hpp>
#include <mgcpp/matrix/device_matrix.hpp>

#include "../cpu_matrix.hpp"
#include "../mgcpp_test.hpp"

template <typename Type>
class device_matrix_test : public ::testing::Test {
 private:
  virtual void SetUp() {
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat.has_value());
  }

 public:
  /* the method name is the test case name */
  void default_constructor();

  void dimension_constructor();

  void dimension_initializing_constructor();

  void third_party_matrix_construction();

  void matrix_init_from_host_data();

  void matrix_init_from_init_list();

  void copy_construction();

  void reallocation_during_copy_assign();

  void no_reallocation_during_copy_assign();

  void copy_to_host();

  void move_constructor();

  void move_assign_operator();

  void matrix_resize();

  void matrix_resize_init();

  void matrix_zero_after_allocation();

  void matrix_zero_without_allocation_failure();
};

using complex_double_type = device_matrix_test<std::complex<double>>;
using complex_type = device_matrix_test<std::complex<float>>;
using double_type = device_matrix_test<double>;
using float_type = device_matrix_test<float>;
using half_type = device_matrix_test<mgcpp::half>;

/* like this we can automatically generate test code for a certain type      */
/* by specializing templates, we can also specialize tests for certain types */

template <typename Type>
void device_matrix_test<Type>::default_constructor() {
  mgcpp::device_matrix<Type, 0> mat;

  auto shape = mat.shape();
  EXPECT_EQ(shape[0], 0);
  EXPECT_EQ(shape[1], 0);
  EXPECT_EQ(mat.data(), nullptr);
  EXPECT_EQ(mat.context(), mgcpp::device_matrix<Type>().context());
}

MGCPP_TEST(complex_double_type, default_constructor)
MGCPP_TEST(complex_type, default_constructor)
MGCPP_TEST(double_type, default_constructor)
MGCPP_TEST(float_type, default_constructor)
MGCPP_TEST(half_type, default_constructor)

template <typename Type>
void device_matrix_test<Type>::dimension_constructor() {
  auto before = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(before.has_value());
  auto before_memory = before.value().first;

  size_t row_dim = 10;
  size_t col_dim = 5;
  mgcpp::device_matrix<Type, 0> mat({row_dim, col_dim});

  auto after = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(after.has_value());
  auto after_memory = after.value().first;

  EXPECT_GT(before_memory, after_memory);

  auto shape = mat.shape();
  EXPECT_EQ(shape[0], row_dim);
  EXPECT_EQ(shape[1], col_dim);
  EXPECT_NE(mat.data(), nullptr);
}

MGCPP_TEST(complex_double_type, dimension_constructor)
MGCPP_TEST(complex_type, dimension_constructor)
MGCPP_TEST(double_type, dimension_constructor)
MGCPP_TEST(float_type, dimension_constructor)
MGCPP_TEST(half_type, dimension_constructor)

template <typename Type>
void device_matrix_test<Type>::dimension_initializing_constructor() {
  auto before = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(before.has_value());
  auto before_memory = before.value().first;

  size_t row_dim = 5;
  size_t col_dim = 10;
  Type init_val = static_cast<Type>(7);
  mgcpp::device_matrix<Type> mat({row_dim, col_dim}, init_val);

  auto after = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(after.has_value());
  auto after_memory = after.value().first;

  EXPECT_GT(before_memory, after_memory);

  auto shape = mat.shape();
  EXPECT_EQ(shape[0], row_dim);
  EXPECT_EQ(shape[1], col_dim);

  EXPECT_NO_THROW(do {
    for (size_t i = 0; i < row_dim; ++i) {
      for (size_t j = 0; j < col_dim; ++j) {
        EXPECT_EQ(mat.check_value(i, j), init_val)
            << "index i: " << i << " j: " << j;
      }
    }
  } while (false););
}

template <>
void device_matrix_test<mgcpp::half>::dimension_initializing_constructor() {
  using half_float::half_cast;

  auto before = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(before.has_value());
  auto before_memory = before.value().first;

  size_t row_dim = 5;
  size_t col_dim = 10;
  mgcpp::half init_val = static_cast<mgcpp::half>(7);
  mgcpp::device_matrix<mgcpp::half> mat({row_dim, col_dim}, init_val);

  auto after = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(after.has_value());
  auto after_memory = after.value().first;

  EXPECT_GT(before_memory, after_memory);

  auto shape = mat.shape();
  EXPECT_EQ(shape[0], row_dim);
  EXPECT_EQ(shape[1], col_dim);

  EXPECT_NO_THROW(do {
    for (size_t i = 0; i < row_dim; ++i) {
      for (size_t j = 0; j < col_dim; ++j) {
        EXPECT_EQ(half_cast<float>(mat.check_value(i, j)),
                  half_cast<float>(init_val))
            << "index i: " << i << " j: " << j;
      }
    }
  } while (false););
}

MGCPP_TEST(complex_double_type, dimension_initializing_constructor)
MGCPP_TEST(complex_type, dimension_initializing_constructor)
MGCPP_TEST(double_type, dimension_initializing_constructor)
MGCPP_TEST(float_type, dimension_initializing_constructor)
MGCPP_TEST(half_type, dimension_initializing_constructor)

template <typename Type>
void device_matrix_test<Type>::third_party_matrix_construction() {
  size_t row_dim = 5;
  size_t col_dim = 10;
  cpu_matrix<Type> host(row_dim, col_dim);

  Type counter = static_cast<Type>(0);
  for (size_t i = 0; i < row_dim; ++i) {
    for (size_t j = 0; j < col_dim; ++j) {
      host(i, j) = counter;
      ++counter;
    }
  }

  EXPECT_NO_THROW(do {
    mgcpp::device_matrix<Type> device(host);

    counter = 0;
    for (size_t i = 0; i < row_dim; ++i) {
      for (size_t j = 0; j < col_dim; ++j) {
        EXPECT_EQ(device.check_value(i, j), counter);
        ++counter;
      }
    }
    EXPECT_EQ(device.shape(), host.shape());
  } while (false));
}

template <>
void device_matrix_test<mgcpp::half>::third_party_matrix_construction() {
  using half_float::half_cast;

  size_t row_dim = 5;
  size_t col_dim = 10;
  cpu_matrix<mgcpp::half> host(row_dim, col_dim);

  mgcpp::half counter = static_cast<mgcpp::half>(0);
  for (size_t i = 0; i < row_dim; ++i) {
    for (size_t j = 0; j < col_dim; ++j) {
      host(i, j) = counter;
      ++counter;
    }
  }

  EXPECT_NO_THROW(do {
    mgcpp::device_matrix<mgcpp::half> device(host);

    counter = 0;
    for (size_t i = 0; i < row_dim; ++i) {
      for (size_t j = 0; j < col_dim; ++j) {
        EXPECT_EQ(half_cast<float>(device.check_value(i, j)),
                  half_cast<float>(counter));
        ++counter;
      }
    }
    EXPECT_EQ(device.shape(), host.shape());
  } while (false));
}

// MGCPP_TEST(complex_double_type, third_party_matrix_construction)
// MGCPP_TEST(complex_type, third_party_matrix_construction)
MGCPP_TEST(double_type, third_party_matrix_construction)
MGCPP_TEST(float_type, third_party_matrix_construction)
MGCPP_TEST(half_type, third_party_matrix_construction)

template <typename Type>
void device_matrix_test<Type>::matrix_init_from_host_data() {
  size_t row_dim = 10;
  size_t col_dim = 10;
  Type* data = (Type*)malloc(sizeof(Type) * row_dim * col_dim);

  Type counter = static_cast<Type>(0);
  for (size_t i = 0; i < row_dim; ++i) {
    for (size_t j = 0; j < col_dim; ++j) {
      data[encode_index(i, j, row_dim)] = counter;
      ++counter;
    }
  }

  auto before = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(before.has_value());
  auto before_freemem = before.value().first;

  mgcpp::device_matrix<Type> mat{};
  EXPECT_NO_THROW(mat = mgcpp::device_matrix<Type>({row_dim, col_dim}, data));

  auto after = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(after.has_value());
  auto after_freemem = after.value().first;

  EXPECT_GT(before_freemem, after_freemem);

  counter = 0;
  EXPECT_NO_THROW(do {
    for (size_t i = 0; i < row_dim; ++i) {
      for (size_t j = 0; j < col_dim; ++j) {
        EXPECT_EQ(mat.check_value(i, j), counter)
            << "index i: " << i << " j: " << j;
        ++counter;
      }
    }
  } while (false););
  free(data);
}

template <>
void device_matrix_test<mgcpp::half>::matrix_init_from_host_data() {
  using half_float::half_cast;

  size_t row_dim = 10;
  size_t col_dim = 10;
  mgcpp::half* data =
      (mgcpp::half*)malloc(sizeof(mgcpp::half) * row_dim * col_dim);

  mgcpp::half counter = static_cast<mgcpp::half>(0);
  for (size_t i = 0; i < row_dim; ++i) {
    for (size_t j = 0; j < col_dim; ++j) {
      data[encode_index(i, j, row_dim)] = counter;
      ++counter;
    }
  }

  auto before = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(before.has_value());
  auto before_freemem = before.value().first;

  mgcpp::device_matrix<mgcpp::half> mat{};
  EXPECT_NO_THROW(
      mat = mgcpp::device_matrix<mgcpp::half>({row_dim, col_dim}, data));

  auto after = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(after.has_value());
  auto after_freemem = after.value().first;

  EXPECT_GT(before_freemem, after_freemem);

  counter = 0;
  EXPECT_NO_THROW(do {
    for (size_t i = 0; i < row_dim; ++i) {
      for (size_t j = 0; j < col_dim; ++j) {
        EXPECT_EQ(half_cast<float>(mat.check_value(i, j)),
                  half_cast<float>(counter))
            << "index i: " << i << " j: " << j;
        ++counter;
      }
    }
  } while (false););
  free(data);
}

// MGCPP_TEST(complex_double_type, matrix_init_from_host_data)
// MGCPP_TEST(complex_type, matrix_init_from_host_data)
MGCPP_TEST(double_type, matrix_init_from_host_data)
MGCPP_TEST(float_type, matrix_init_from_host_data)
MGCPP_TEST(half_type, matrix_init_from_host_data)

template <typename Type>
void device_matrix_test<Type>::matrix_init_from_init_list() {
  auto init_list = std::initializer_list<std::initializer_list<Type>>{
      {0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}, {6.0f, 7.0f, 8.0f}};

  auto before = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(before.has_value());
  auto before_freemem = before.value().first;

  mgcpp::device_matrix<Type> mat{};
  EXPECT_NO_THROW(mat = mgcpp::device_matrix<Type>::from_list(init_list));

  auto after = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(after.has_value());
  auto after_freemem = after.value().first;

  EXPECT_GT(before_freemem, after_freemem);

  EXPECT_NO_THROW(do {
    EXPECT_EQ(0.0f, mat.check_value(0, 0));
    EXPECT_EQ(1.0f, mat.check_value(0, 1));
    EXPECT_EQ(2.0f, mat.check_value(0, 2));
    EXPECT_EQ(3.0f, mat.check_value(1, 0));
    EXPECT_EQ(4.0f, mat.check_value(1, 1));
    EXPECT_EQ(5.0f, mat.check_value(1, 2));
    EXPECT_EQ(6.0f, mat.check_value(2, 0));
    EXPECT_EQ(7.0f, mat.check_value(2, 1));
    EXPECT_EQ(8.0f, mat.check_value(2, 2));
  } while (false));
}

// MGCPP_TEST(complex_double_type, matrix_init_from_init_list)
// MGCPP_TEST(complex_type, matrix_init_from_init_list)
MGCPP_TEST(double_type, matrix_init_from_init_list)
MGCPP_TEST(float_type, matrix_init_from_init_list)
// MGCPP_TEST(half_type, matrix_init_from_init_list)

template <typename Type>
void device_matrix_test<Type>::copy_construction() {
  size_t row_dim = 5;
  size_t col_dim = 10;
  Type init = static_cast<Type>(7);

  mgcpp::device_matrix<Type> original({row_dim, col_dim}, init);
  mgcpp::device_matrix<Type> copied{};
  EXPECT_NO_THROW(copied = mgcpp::device_matrix<Type>(original));

  EXPECT_NO_THROW(do {
    for (auto i = 0u; i < row_dim; ++i) {
      for (auto j = 0u; j < col_dim; ++j) {
        EXPECT_EQ(original.check_value(i, j), copied.check_value(i, j));
      }
    }
  } while (false));

  EXPECT_EQ(original.shape(), copied.shape());
}

template <>
void device_matrix_test<mgcpp::half>::copy_construction() {
  using half_float::half_cast;
  size_t row_dim = 5;
  size_t col_dim = 10;
  mgcpp::half init = static_cast<mgcpp::half>(7);

  mgcpp::device_matrix<mgcpp::half> original({row_dim, col_dim}, init);
  mgcpp::device_matrix<mgcpp::half> copied{};
  EXPECT_NO_THROW(copied = mgcpp::device_matrix<mgcpp::half>(original));

  EXPECT_NO_THROW(do {
    for (auto i = 0u; i < row_dim; ++i) {
      for (auto j = 0u; j < col_dim; ++j) {
        EXPECT_EQ(half_cast<float>(original.check_value(i, j)),
                  half_cast<float>(copied.check_value(i, j)));
      }
    }
  } while (false));

  EXPECT_EQ(original.shape(), copied.shape());
}

MGCPP_TEST(complex_double_type, copy_construction)
MGCPP_TEST(complex_type, copy_construction)
MGCPP_TEST(double_type, copy_construction)
MGCPP_TEST(float_type, copy_construction)
MGCPP_TEST(half_type, copy_construction)

template <typename Type>
void device_matrix_test<Type>::reallocation_during_copy_assign() {
  size_t row_dim = 5;
  size_t col_dim = 10;
  Type init = static_cast<Type>(7);

  mgcpp::device_matrix<Type> original({row_dim, col_dim}, init);
  mgcpp::device_matrix<Type> copied({row_dim / 2, col_dim / 2});
  size_t before_capacity = copied.capacity();

  EXPECT_NO_THROW(copied = original);

  EXPECT_EQ(copied.check_value(0, 0), init);  // supressing optimization

  EXPECT_EQ(original.shape(), copied.shape());
  EXPECT_EQ(original.capacity(), copied.capacity());
  EXPECT_LT(before_capacity, copied.capacity());
}

template <>
void device_matrix_test<mgcpp::half>::reallocation_during_copy_assign() {
  using half_float::half_cast;
  size_t row_dim = 5;
  size_t col_dim = 10;
  mgcpp::half init = static_cast<mgcpp::half>(7);

  mgcpp::device_matrix<mgcpp::half> original({row_dim, col_dim}, init);
  mgcpp::device_matrix<mgcpp::half> copied({row_dim / 2, col_dim / 2});
  size_t before_capacity = copied.capacity();

  EXPECT_NO_THROW(copied = original);

  EXPECT_EQ(half_cast<float>(copied.check_value(0, 0)),
            half_cast<float>(init));  // supressing optimization

  EXPECT_EQ(original.shape(), copied.shape());
  EXPECT_EQ(original.capacity(), copied.capacity());
  EXPECT_LT(before_capacity, copied.capacity());
}

MGCPP_TEST(complex_double_type, reallocation_during_copy_assign)
MGCPP_TEST(complex_type, reallocation_during_copy_assign)
MGCPP_TEST(double_type, reallocation_during_copy_assign)
MGCPP_TEST(float_type, reallocation_during_copy_assign)
MGCPP_TEST(half_type, reallocation_during_copy_assign)

template <typename Type>
void device_matrix_test<Type>::no_reallocation_during_copy_assign() {
  size_t row_dim = 5;
  size_t col_dim = 10;
  Type init = 7;

  auto before = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(before.has_value());
  auto before_freemem = before.value().first;

  mgcpp::device_matrix<Type> original({row_dim, col_dim}, init);
  mgcpp::device_matrix<Type> copied({row_dim * 2, col_dim * 2});
  EXPECT_NO_THROW(copied = original);

  EXPECT_EQ(copied.check_value(0, 0), init);  // supressing optimization

  auto after = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(after.has_value());
  auto after_freemem = before.value().first;

  EXPECT_EQ(after_freemem, before_freemem);

  EXPECT_EQ(original.shape(), copied.shape());
  EXPECT_LT(original.capacity(), copied.capacity());
}

template <>
void device_matrix_test<mgcpp::half>::no_reallocation_during_copy_assign() {
  using half_float::half_cast;
  size_t row_dim = 5;
  size_t col_dim = 10;
  mgcpp::half init = static_cast<mgcpp::half>(7);

  auto before = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(before.has_value());
  auto before_freemem = before.value().first;

  mgcpp::device_matrix<mgcpp::half> original({row_dim, col_dim}, init);
  mgcpp::device_matrix<mgcpp::half> copied({row_dim * 2, col_dim * 2});
  EXPECT_NO_THROW(copied = original);

  EXPECT_EQ(half_cast<float>(copied.check_value(0, 0)),
            half_cast<float>(init));  // supressing optimization

  auto after = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(after.has_value());
  auto after_freemem = before.value().first;

  EXPECT_EQ(after_freemem, before_freemem);

  EXPECT_EQ(original.shape(), copied.shape());
  EXPECT_LT(original.capacity(), copied.capacity());
}

MGCPP_TEST(complex_double_type, no_reallocation_during_copy_assign)
MGCPP_TEST(complex_type, no_reallocation_during_copy_assign)
MGCPP_TEST(double_type, no_reallocation_during_copy_assign)
MGCPP_TEST(float_type, no_reallocation_during_copy_assign)
MGCPP_TEST(half_type, no_reallocation_during_copy_assign)

template <typename Type>
void device_matrix_test<Type>::copy_to_host() {
  size_t row_dim = 5;
  size_t col_dim = 10;
  Type init = static_cast<Type>(7);
  mgcpp::device_matrix<Type> mat({row_dim, col_dim}, init);

  Type* host = (Type*)malloc(sizeof(Type) * row_dim * col_dim);
  EXPECT_NO_THROW(mat.copy_to_host(host));

  for (size_t i = 0; i < row_dim; ++i) {
    for (size_t j = 0; j < row_dim; ++j) {
      EXPECT_EQ(host[encode_index(i, j, col_dim)], init);
    }
  }
  free(host);
}

template <>
void device_matrix_test<mgcpp::half>::copy_to_host() {
  using half_float::half_cast;
  size_t row_dim = 5;
  size_t col_dim = 10;
  mgcpp::half init = static_cast<mgcpp::half>(7);
  mgcpp::device_matrix<mgcpp::half> mat({row_dim, col_dim}, init);

  mgcpp::half* host =
      (mgcpp::half*)malloc(sizeof(mgcpp::half) * row_dim * col_dim);
  EXPECT_NO_THROW(mat.copy_to_host(host));

  for (size_t i = 0; i < row_dim; ++i) {
    for (size_t j = 0; j < row_dim; ++j) {
      EXPECT_EQ(half_cast<float>(host[encode_index(i, j, col_dim)]),
                half_cast<float>(init));
    }
  }
  free(host);
}

MGCPP_TEST(complex_double_type, copy_to_host)
MGCPP_TEST(complex_type, copy_to_host)
MGCPP_TEST(double_type, copy_to_host)
MGCPP_TEST(float_type, copy_to_host)
MGCPP_TEST(half_type, copy_to_host)

template <typename Type>
void device_matrix_test<Type>::move_constructor() {
  size_t row_dim = 5;
  size_t col_dim = 10;
  Type init = static_cast<Type>(7);

  mgcpp::device_matrix<Type> original({row_dim, col_dim}, init);

  auto before = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(before.has_value());
  auto before_freemem = before.value().first;

  mgcpp::device_matrix<Type> moved(std::move(original));

  auto after = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(after.has_value());
  auto after_freemem = after.value().first;

  EXPECT_EQ(before_freemem, after_freemem);

  EXPECT_NO_THROW(do {
    for (auto i = 0u; i < row_dim; ++i) {
      for (auto j = 0u; j < col_dim; ++j) {
        EXPECT_EQ(moved.check_value(i, j), Type(7));
      }
    }
  } while (false));

  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(moved.shape()[0], row_dim);
  EXPECT_EQ(moved.shape()[1], col_dim);
}

template <>
void device_matrix_test<mgcpp::half>::move_constructor() {
  using half_float::half_cast;
  size_t row_dim = 5;
  size_t col_dim = 10;
  mgcpp::half init = static_cast<mgcpp::half>(7);

  mgcpp::device_matrix<mgcpp::half> original({row_dim, col_dim}, init);

  auto before = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(before.has_value());
  auto before_freemem = before.value().first;

  mgcpp::device_matrix<mgcpp::half> moved(std::move(original));

  auto after = mgcpp::cuda_mem_get_info();
  EXPECT_TRUE(after.has_value());
  auto after_freemem = after.value().first;

  EXPECT_EQ(before_freemem, after_freemem);

  EXPECT_NO_THROW(do {
    for (auto i = 0u; i < row_dim; ++i) {
      for (auto j = 0u; j < col_dim; ++j) {
        EXPECT_EQ(half_cast<float>(moved.check_value(i, j)), 7.0f);
      }
    }
  } while (false));

  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(moved.shape()[0], row_dim);
  EXPECT_EQ(moved.shape()[1], col_dim);
}

MGCPP_TEST(complex_double_type, move_constructor)
MGCPP_TEST(complex_type, move_constructor)
MGCPP_TEST(double_type, move_constructor)
MGCPP_TEST(float_type, move_constructor)
MGCPP_TEST(half_type, move_constructor)

template <typename Type>
void device_matrix_test<Type>::move_assign_operator() {
  size_t row_dim = 5;
  size_t col_dim = 10;
  Type init = static_cast<Type>(7);

  mgcpp::device_matrix<Type> original({row_dim, col_dim}, init);
  mgcpp::device_matrix<Type> moved({row_dim * 2, col_dim * 2});
  size_t before_capacity = moved.capacity();

  moved = std::move(original);

  EXPECT_GT(before_capacity, moved.capacity());

  EXPECT_NO_THROW(do {
    for (auto i = 0u; i < row_dim; ++i) {
      for (auto j = 0u; j < col_dim; ++j) {
        EXPECT_EQ(moved.check_value(i, j), init);
      }
    }
  } while (false));

  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(moved.shape()[0], row_dim);
  EXPECT_EQ(moved.shape()[1], col_dim);
}

template <>
void device_matrix_test<mgcpp::half>::move_assign_operator() {
  using half_float::half_cast;
  size_t row_dim = 5;
  size_t col_dim = 10;
  mgcpp::half init = static_cast<mgcpp::half>(7);

  mgcpp::device_matrix<mgcpp::half> original({row_dim, col_dim}, init);
  mgcpp::device_matrix<mgcpp::half> moved({row_dim * 2, col_dim * 2});
  size_t before_capacity = moved.capacity();

  moved = std::move(original);

  EXPECT_GT(before_capacity, moved.capacity());

  EXPECT_NO_THROW(do {
    for (auto i = 0u; i < row_dim; ++i) {
      for (auto j = 0u; j < col_dim; ++j) {
        EXPECT_EQ(half_cast<float>(moved.check_value(i, j)),
                  half_cast<float>(init));
      }
    }
  } while (false));

  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(moved.shape()[0], row_dim);
  EXPECT_EQ(moved.shape()[1], col_dim);
}

MGCPP_TEST(complex_double_type, move_assign_operator)
MGCPP_TEST(complex_type, move_assign_operator)
MGCPP_TEST(double_type, move_assign_operator)
MGCPP_TEST(float_type, move_assign_operator)
MGCPP_TEST(half_type, move_assign_operator)

template <typename Type>
void device_matrix_test<Type>::matrix_resize() {
  size_t row_dim = 10;
  size_t col_dim = 10;
  mgcpp::device_matrix<Type> mat({row_dim, col_dim});
  size_t before_capacity = mat.capacity();

  EXPECT_NO_THROW(mat.resize({row_dim * 2, col_dim * 2}));

  EXPECT_GT(mat.capacity(), before_capacity);
}

MGCPP_TEST(complex_double_type, matrix_resize)
MGCPP_TEST(complex_type, matrix_resize)
MGCPP_TEST(double_type, matrix_resize)
MGCPP_TEST(float_type, matrix_resize)
MGCPP_TEST(half_type, matrix_resize)

template <typename Type>
void device_matrix_test<Type>::matrix_resize_init() {
  size_t row_dim = 10;
  size_t col_dim = 10;
  mgcpp::device_matrix<Type> mat({row_dim, col_dim});
  size_t before_capacity = mat.capacity();

  Type init_val = static_cast<Type>(7);

  EXPECT_NO_THROW(mat.resize({row_dim * 2, col_dim * 2}, init_val));

  EXPECT_GT(mat.capacity(), before_capacity);

  EXPECT_NO_THROW(do {
    for (size_t i = 0; i < row_dim * 2; ++i) {
      for (size_t j = 0; j < col_dim * 2; ++j) {
        EXPECT_EQ(mat.check_value(i, j), init_val)
            << "index i: " << i << " j: " << j;
      }
    }
  } while (false););
}

template <>
void device_matrix_test<mgcpp::half>::matrix_resize_init() {
  using half_float::half_cast;
  size_t row_dim = 10;
  size_t col_dim = 10;
  mgcpp::device_matrix<mgcpp::half> mat({row_dim, col_dim});
  size_t before_capacity = mat.capacity();

  mgcpp::half init_val = static_cast<mgcpp::half>(7);

  EXPECT_NO_THROW(mat.resize({row_dim * 2, col_dim * 2}, init_val));

  EXPECT_GT(mat.capacity(), before_capacity);

  EXPECT_NO_THROW(do {
    for (size_t i = 0; i < row_dim * 2; ++i) {
      for (size_t j = 0; j < col_dim * 2; ++j) {
        EXPECT_EQ(half_cast<float>(mat.check_value(i, j)),
                  half_cast<float>(init_val))
            << "index i: " << i << " j: " << j;
      }
    }
  } while (false););
}

MGCPP_TEST(complex_double_type, matrix_resize_init)
MGCPP_TEST(complex_type, matrix_resize_init)
MGCPP_TEST(double_type, matrix_resize_init)
MGCPP_TEST(float_type, matrix_resize_init)
MGCPP_TEST(half_type, matrix_resize_init)

template <typename Type>
void device_matrix_test<Type>::matrix_zero_after_allocation() {
  size_t row_dim = 5;
  size_t col_dim = 10;
  mgcpp::device_matrix<Type> mat({row_dim, col_dim});
  EXPECT_NO_THROW(mat.zero());

  EXPECT_NO_THROW(do {
    for (size_t i = 0; i < row_dim; ++i) {
      for (size_t j = 0; j < col_dim; ++j) {
        EXPECT_EQ(mat.check_value(i, j), 0) << "index i: " << i << " j: " << j;
      }
    }
  } while (false););
}

// template<>
// void
// device_matrix_test<mgcpp::half>::
// matrix_zero_after_allocation()
// {
//     using half_float::half_cast;
//     size_t row_dim = 5;
//     size_t col_dim = 10;
//     mgcpp::device_matrix<mgcpp::half> mat({row_dim, col_dim});
//     EXPECT_NO_THROW(mat.zero());

//     EXPECT_NO_THROW(do {
//             for (size_t i = 0; i < row_dim; ++i) {
//                 for (size_t j = 0; j < col_dim; ++j) {
//                     EXPECT_EQ(half_cast<float>(mat.check_value(i, j)),
//                               0.0f) << "index i: " << i << " j: " << j;
//                 }
//             }

//         } while (false););
// }

// MGCPP_TEST(complex_double_type, matrix_zero_after_allocation)
// MGCPP_TEST(complex_type, matrix_zero_after_allocation)
MGCPP_TEST(double_type, matrix_zero_after_allocation)
MGCPP_TEST(float_type, matrix_zero_after_allocation)
// MGCPP_TEST(half_type, matrix_zero_after_allocation)

template <typename Type>
void device_matrix_test<Type>::matrix_zero_without_allocation_failure() {
  mgcpp::device_matrix<float> mat{};
  EXPECT_ANY_THROW(mat.zero());
}

MGCPP_TEST(complex_double_type, matrix_zero_without_allocation_failure)
MGCPP_TEST(complex_type, matrix_zero_without_allocation_failure)
MGCPP_TEST(double_type, matrix_zero_without_allocation_failure)
MGCPP_TEST(float_type, matrix_zero_without_allocation_failure)
MGCPP_TEST(half_type, matrix_zero_without_allocation_failure)
