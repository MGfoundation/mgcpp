
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#define ERROR_CHECK_EXCEPTION true

#include "../cpu_matrix.hpp"
#include "../mgcpp_test.hpp"
#include "../test_utils.hpp"

#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/operations/fft.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <complex>
#include <random>
#include <valarray>

#define TEST_SIZE 2048

template <typename T>
using carray = std::valarray<std::complex<T>>;

template <typename T>
using cmat = std::valarray<carray<T>>;

constexpr double PI = 3.1415926535897932384626433832795028;
template <typename T>
void fft(carray<T>& a, bool inv) {
  int n = a.size();
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    while (!((j ^= bit) & bit))
      bit >>= 1;
    if (i < j)
      std::swap(a[i], a[j]);
  }
  for (int i = 1; i < n; i <<= 1) {
    double x = inv ? PI / i : -PI / i;
    auto w = std::polar<T>(1., x);
    for (int j = 0; j < n; j += i << 1) {
      std::complex<T> th = {1, 0};
      for (int k = 0; k < i; k++) {
        auto tmp = a[i + j + k] * th;
        a[i + j + k] = a[j + k] - tmp;
        a[j + k] += tmp;
        th *= w;
      }
    }
  }
  if (inv) {
    a /= std::complex<T>(n);
  }
}

template <typename T>
void fft(cmat<T>& c, int dir) {
  int nx = c.size(), ny = c[0].size();
  carray<T> R(nx);
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++)
      R[i] = c[i][j];
    fft(R, dir);
    for (int i = 0; i < nx; i++)
      c[i][j] = R[i];
  }
  for (int i = 0; i < nx; i++)
    fft(c[i], dir);
}

template <typename Type>
class fft_operation_test : public ::testing::Test {
 private:
  virtual void SetUp() {
    auto set_device_stat = mgcpp::cuda_set_device(0);
    EXPECT_TRUE(set_device_stat.has_value());
  }

 public:
  /* the method name is the test case name */
  void real_to_complex_fwd_fft();

  void complex_to_real_fwd_fft();

  void real_complex_real_roundtrip_2d();

  void upsampling_fft_even();

  void complex_to_complex_fwd_fft();

  void float_complex_to_complex_inv_fft();
};

using fft_complex_double_type = fft_operation_test<std::complex<double>>;
using fft_complex_type = fft_operation_test<std::complex<float>>;
using fft_double_type = fft_operation_test<double>;
using fft_float_type = fft_operation_test<float>;
#ifdef USE_HALF
using fft_half_type = fft_operation_test<mgcpp::half>;
#endif

template <typename Type>
void fft_operation_test<Type>::real_to_complex_fwd_fft() {
  for (size_t size = 1; size <= TEST_SIZE; size *= 2) {
    auto vec = mgcpp::device_vector<Type>(size);
    random_vector(vec);

    carray<Type> expected(size);
    for (auto i = 0u; i < vec.size(); ++i)
      expected[i] = {vec.check_value(i), 0};
    fft(expected, false);

    mgcpp::device_vector<mgcpp::complex<Type>> result;
    EXPECT_NO_THROW({ result = mgcpp::strict::rfft(vec); });

    EXPECT_EQ(result.size(), size / 2 + 1);
    for (auto i = 0u; i < result.size(); ++i) {
      EXPECT_NEAR(result.check_value(i).real(), expected[i].real(), 1e-3)
          << "size = " << size << ", i = " << i;
      EXPECT_NEAR(result.check_value(i).imag(), expected[i].imag(), 1e-3)
          << "size = " << size << ", i = " << i;
    }
  }
}

MGCPP_TEST(fft_float_type, real_to_complex_fwd_fft)
MGCPP_TEST(fft_double_type, real_to_complex_fwd_fft)
// MGCPP_TEST(half_type, real_to_complex_fwd_fft)

template <typename Type>
void fft_operation_test<Type>::complex_to_real_fwd_fft() {
  for (size_t size = 2; size <= TEST_SIZE; size *= 2) {
    auto vec = mgcpp::device_vector<mgcpp::complex<Type>>(size / 2 + 1);
    random_vector(vec);

    carray<float> expected(size);
    for (auto i = 0u; i < vec.size(); ++i) {
      expected[i] = vec.check_value(i);
    }
    for (auto i = vec.size(); i < size; ++i) {
      auto c = vec.check_value(size - i);
      expected[i] = std::conj(c);
    }
    fft(expected, true);

    mgcpp::device_vector<Type> result;
    EXPECT_NO_THROW({ result = mgcpp::strict::irfft(vec); });

    EXPECT_EQ(result.size(), size);
    for (auto i = 0u; i < result.size(); ++i) {
      EXPECT_NEAR(result.check_value(i), expected[i].real(), 1e-3)
          << "size = " << size << ", i = " << i;
    }
  }
}

MGCPP_TEST(fft_float_type, complex_to_real_fwd_fft)
MGCPP_TEST(fft_double_type, complex_to_real_fwd_fft)

template <typename Type>
void fft_operation_test<Type>::real_complex_real_roundtrip_2d() {
  auto shape = mgcpp::make_shape(32, 16);
  auto mat = mgcpp::device_matrix<mgcpp::complex<Type>>(shape);
  random_matrix(mat);

  mgcpp::device_matrix<mgcpp::complex<Type>> F;
  EXPECT_NO_THROW({ F = mgcpp::strict::rfft(mat); });

  mgcpp::device_matrix<Type> result;
  EXPECT_NO_THROW({ result = mgcpp::strict::irfft(F); });

  EXPECT_EQ(result.shape(), mat.shape());
  for (auto i = 0u; i < result.shape()[0]; ++i) {
    for (auto j = 0u; j < result.shape()[1]; ++j) {
      EXPECT_NEAR(result.check_value(i, j), mat.check_value(i, j), 1e-3)
          << "size = (" << shape[0] << ", " << shape[1] << "), i = " << i
          << ", j = " << j;
    }
  }
}

// MGCPP_TEST(float_type, real_complex_real_roundtrip_2d)
// MGCPP_TEST(double_type, real_complex_real_roundtrip_2d)

template <typename Type>
void fft_operation_test<Type>::upsampling_fft_even() {
  size_t size = 16, new_size = 32;
  mgcpp::device_vector<mgcpp::complex<Type>> vec(size / 2 + 1);
  random_vector(vec);

  carray<double> expected(new_size);
  for (auto i = 0u; i < vec.size(); ++i) {
    expected[i] = vec.check_value(i);
  }
  for (auto i = new_size - vec.size() + 1; i < new_size; ++i) {
    expected[i] = std::conj(expected[new_size - i]);
  }
  fft(expected, true);

  mgcpp::device_vector<Type> result;
  EXPECT_NO_THROW({ result = mgcpp::strict::irfft(vec, new_size); });

  EXPECT_EQ(result.size(), new_size);
  for (auto i = 0u; i < result.size(); ++i) {
    EXPECT_NEAR(result.check_value(i), expected[i].real(), 1e-5)
        << "size = " << size << ", i = " << i;
  }
}

MGCPP_TEST(fft_float_type, upsampling_fft_even)
MGCPP_TEST(fft_double_type, upsampling_fft_even)

// forward cfft
template <typename Type>
void fft_operation_test<Type>::complex_to_complex_fwd_fft() {
  for (size_t size = 1; size <= TEST_SIZE; size *= 2) {
    mgcpp::device_vector<mgcpp::complex<Type>> vec(size);
    random_vector(vec);

    carray<Type> expected(size);
    for (auto i = 0u; i < vec.size(); ++i) {
      expected[i] = vec.check_value(i);
    }
    fft(expected, false);

    mgcpp::device_vector<mgcpp::complex<Type>> result;
    EXPECT_NO_THROW(
        { result = mgcpp::strict::cfft(vec, mgcpp::fft_direction::forward); });

    EXPECT_EQ(result.size(), size);
    for (auto i = 0u; i < result.size(); ++i) {
      EXPECT_NEAR(result.check_value(i).real(), expected[i].real(), 1e-3)
          << "size = " << size << ", i = " << i;
      EXPECT_NEAR(result.check_value(i).imag(), expected[i].imag(), 1e-3)
          << "size = " << size << ", i = " << i;
    }
  }
}

MGCPP_TEST(fft_float_type, complex_to_complex_fwd_fft)
MGCPP_TEST(fft_double_type, complex_to_complex_fwd_fft)

template <typename Type>
void fft_operation_test<Type>::float_complex_to_complex_inv_fft() {
  for (size_t size = 1; size <= TEST_SIZE; size *= 2) {
    mgcpp::device_vector<mgcpp::complex<Type>> vec(size);
    random_vector(vec);

    carray<Type> expected(size);
    for (auto i = 0u; i < vec.size(); ++i) {
      expected[i] = vec.check_value(i);
    }
    fft(expected, true);

    mgcpp::device_vector<mgcpp::complex<Type>> result;
    EXPECT_NO_THROW(
        { result = mgcpp::strict::cfft(vec, mgcpp::fft_direction::inverse); });

    EXPECT_EQ(result.size(), size);
    for (auto i = 0u; i < result.size(); ++i) {
      EXPECT_NEAR(result.check_value(i).real(), expected[i].real(), 1e-3)
          << "size = " << size << ", i = " << i;
      EXPECT_NEAR(result.check_value(i).imag(), expected[i].imag(), 1e-3)
          << "size = " << size << ", i = " << i;
    }
  }
}

MGCPP_TEST(fft_float_type, float_complex_to_complex_inv_fft)
MGCPP_TEST(fft_double_type, float_complex_to_complex_inv_fft)
