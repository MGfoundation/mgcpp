
#include <gtest/gtest.h>

#include <mgcpp/operations/fft.hpp>

#define TEST_SIZE 2048

#include <complex>
#include <random>
#include <valarray>
template <typename T>
using carray = std::valarray<std::complex<T>>;
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
using cmat = std::valarray<carray<T>>;
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

std::default_random_engine rng;
std::uniform_real_distribution<double> dist(0.0, 1.0);

// rfft
TEST(fft_operation, float_real_to_complex_fwd_fft) {
  for (size_t size = 1; size <= TEST_SIZE; size *= 2) {
    mgcpp::device_vector<float> vec(size);
    for (auto i = 0u; i < size; ++i)
      vec.set_value(i, dist(rng));

    carray<float> expected(size);
    for (auto i = 0u; i < vec.size(); ++i)
      expected[i] = {vec.check_value(i), 0};
    fft(expected, false);

    mgcpp::device_vector<mgcpp::complex<float>> result;
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

TEST(fft_operation, float_real_to_complex_fwd_fft_2d) {
  auto shape = mgcpp::make_shape(32, 16);
  mgcpp::device_matrix<float> mat(shape);
  for (auto i = 0u; i < shape[0]; ++i)
    for (auto j = 0u; j < shape[1]; ++j)
      mat.set_value(i, j, dist(rng));

  cmat<float> expected(carray<float>(shape[1]), shape[0]);
  for (auto i = 0u; i < mat.shape()[0]; ++i)
    for (auto j = 0u; j < mat.shape()[1]; ++j)
      expected[i][j] = {mat.check_value(i, j), 0};
  fft(expected, false);

  mgcpp::device_matrix<mgcpp::complex<float>> result;
  EXPECT_NO_THROW({ result = mgcpp::strict::rfft(mat); });

  EXPECT_EQ(result.shape()[0], shape[0] / 2 + 1);
  EXPECT_EQ(result.shape()[1], shape[1]);

  for (auto i = 0u; i < result.shape()[0]; ++i) {
    for (auto j = 0u; j < result.shape()[1]; ++j) {
      EXPECT_NEAR(result.check_value(i, j).real(), expected[i][j].real(), 1e-3)
          << "size = (" << shape[0] << ", " << shape[1] << "), i = " << i
          << ", j = " << j;
      EXPECT_NEAR(result.check_value(i, j).imag(), expected[i][j].imag(), 1e-3)
          << "size = (" << shape[0] << ", " << shape[1] << "), i = " << i
          << ", j = " << j;
    }
  }
}

TEST(fft_operation, double_real_to_complex_fwd_fft) {
  for (size_t size = 1; size <= TEST_SIZE; size *= 2) {
    mgcpp::device_vector<double> vec(size);
    for (auto i = 0u; i < size; ++i)
      vec.set_value(i, dist(rng));

    carray<double> expected(size);
    for (auto i = 0u; i < vec.size(); ++i)
      expected[i] = {vec.check_value(i), 0};
    fft(expected, false);

    mgcpp::device_vector<mgcpp::complex<double>> result;
    EXPECT_NO_THROW({ result = mgcpp::strict::rfft(vec); });

    EXPECT_EQ(result.size(), size / 2 + 1);
    for (auto i = 0u; i < result.size(); ++i) {
      EXPECT_NEAR(result.check_value(i).real(), expected[i].real(), 1e-5)
          << "size = " << size << ", i = " << i;
      EXPECT_NEAR(result.check_value(i).imag(), expected[i].imag(), 1e-5)
          << "size = " << size << ", i = " << i;
    }
  }
}

// irfft
TEST(fft_operation, float_complex_to_real_inv_fft) {
  for (size_t size = 2; size <= TEST_SIZE; size *= 2) {
    mgcpp::device_vector<mgcpp::complex<float>> vec(size / 2 + 1);
    for (auto i = 0u; i < vec.size(); ++i) {
      std::complex<float> random_complex(dist(rng), dist(rng));
      vec.set_value(i, random_complex);
    }

    carray<float> expected(size);
    for (auto i = 0u; i < vec.size(); ++i) {
      expected[i] = vec.check_value(i);
    }
    for (auto i = vec.size(); i < size; ++i) {
      auto c = vec.check_value(size - i);
      expected[i] = std::conj(c);
    }
    fft(expected, true);

    mgcpp::device_vector<float> result;
    EXPECT_NO_THROW({ result = mgcpp::strict::irfft(vec); });

    EXPECT_EQ(result.size(), size);
    for (auto i = 0u; i < result.size(); ++i) {
      EXPECT_NEAR(result.check_value(i), expected[i].real(), 1e-3)
          << "size = " << size << ", i = " << i;
    }
  }
}

TEST(fft_operation, float_real_complex_real_roundtrip_2d) {
  auto shape = mgcpp::make_shape(32, 16);
  mgcpp::device_matrix<float> mat(shape);
  for (auto i = 0u; i < shape[0]; ++i)
    for (auto j = 0u; j < shape[1]; ++j)
      mat.set_value(i, j, dist(rng));

  mgcpp::device_matrix<mgcpp::complex<float>> F;
  EXPECT_NO_THROW({ F = mgcpp::strict::rfft(mat); });

  mgcpp::device_matrix<float> result;
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

TEST(fft_operation, double_complex_to_real_inv_fft) {
  for (size_t size = 2; size <= TEST_SIZE; size *= 2) {
    mgcpp::device_vector<mgcpp::complex<double>> vec(size / 2 + 1);
    for (auto i = 0u; i < vec.size(); ++i) {
      std::complex<double> random_complex(dist(rng), dist(rng));
      vec.set_value(i, random_complex);
    }

    carray<double> expected(size);
    for (auto i = 0u; i < vec.size(); ++i) {
      expected[i] = vec.check_value(i);
    }
    for (auto i = vec.size(); i < size; ++i) {
      auto c = vec.check_value(size - i);
      expected[i] = std::conj(c);
    }
    fft(expected, true);

    mgcpp::device_vector<double> result;
    EXPECT_NO_THROW({ result = mgcpp::strict::irfft(vec); });

    EXPECT_EQ(result.size(), size);
    for (auto i = 0u; i < result.size(); ++i) {
      EXPECT_NEAR(result.check_value(i), expected[i].real(), 1e-5)
          << "size = " << size << ", i = " << i;
    }
  }
}

TEST(fft_operation, double_upsampling_fft_even) {
  size_t size = 16, new_size = 32;
  mgcpp::device_vector<mgcpp::complex<double>> vec(size / 2 + 1);
  for (auto i = 0u; i < vec.size(); ++i) {
    std::complex<double> random_complex(dist(rng), dist(rng));
    vec.set_value(i, random_complex);
  }

  carray<double> expected(new_size);
  for (auto i = 0u; i < vec.size(); ++i) {
    expected[i] = vec.check_value(i);
  }
  for (auto i = new_size - vec.size() + 1; i < new_size; ++i) {
    expected[i] = std::conj(expected[new_size - i]);
  }
  fft(expected, true);

  mgcpp::device_vector<double> result;
  EXPECT_NO_THROW({ result = mgcpp::strict::irfft(vec, new_size); });

  EXPECT_EQ(result.size(), new_size);
  for (auto i = 0u; i < result.size(); ++i) {
    EXPECT_NEAR(result.check_value(i), expected[i].real(), 1e-5)
        << "size = " << size << ", i = " << i;
  }
}

// forward cfft
TEST(fft_operation, float_complex_to_complex_fwd_fft) {
  for (size_t size = 1; size <= TEST_SIZE; size *= 2) {
    mgcpp::device_vector<mgcpp::complex<float>> vec(size);
    for (auto i = 0u; i < size; ++i) {
      std::complex<float> random_complex(dist(rng), dist(rng));
      vec.set_value(i, random_complex);
    }

    carray<float> expected(size);
    for (auto i = 0u; i < vec.size(); ++i) {
      expected[i] = vec.check_value(i);
    }
    fft(expected, false);

    mgcpp::device_vector<mgcpp::complex<float>> result;
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

TEST(fft_operation, double_complex_to_complex_fwd_fft) {
  for (size_t size = 1; size <= TEST_SIZE; size *= 2) {
    mgcpp::device_vector<mgcpp::complex<double>> vec(size);
    for (auto i = 0u; i < size; ++i) {
      std::complex<double> random_complex(dist(rng), dist(rng));
      vec.set_value(i, random_complex);
    }

    carray<double> expected(size);
    for (auto i = 0u; i < vec.size(); ++i) {
      expected[i] = vec.check_value(i);
    }
    fft(expected, false);

    mgcpp::device_vector<mgcpp::complex<double>> result;
    EXPECT_NO_THROW(
        { result = mgcpp::strict::cfft(vec, mgcpp::fft_direction::forward); });

    EXPECT_EQ(result.size(), size);
    for (auto i = 0u; i < result.size(); ++i) {
      EXPECT_NEAR(result.check_value(i).real(), expected[i].real(), 1e-5)
          << "size = " << size << ", i = " << i;
      EXPECT_NEAR(result.check_value(i).imag(), expected[i].imag(), 1e-5)
          << "size = " << size << ", i = " << i;
    }
  }
}

// inverse cfft
TEST(fft_operation, float_complex_to_complex_inv_fft) {
  for (size_t size = 1; size <= TEST_SIZE; size *= 2) {
    mgcpp::device_vector<mgcpp::complex<float>> vec(size);
    for (auto i = 0u; i < size; ++i) {
      std::complex<float> random_complex(dist(rng), dist(rng));
      vec.set_value(i, random_complex);
    }

    carray<float> expected(size);
    for (auto i = 0u; i < vec.size(); ++i) {
      expected[i] = vec.check_value(i);
    }
    fft(expected, true);

    mgcpp::device_vector<mgcpp::complex<float>> result;
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

TEST(fft_operation, double_complex_to_complex_inv_fft) {
  for (size_t size = 1; size <= TEST_SIZE; size *= 2) {
    mgcpp::device_vector<mgcpp::complex<double>> vec(size);
    for (auto i = 0u; i < size; ++i) {
      std::complex<double> random_complex(dist(rng), dist(rng));
      vec.set_value(i, random_complex);
    }

    carray<double> expected(size);
    for (auto i = 0u; i < vec.size(); ++i) {
      expected[i] = vec.check_value(i);
    }
    fft(expected, true);

    mgcpp::device_vector<mgcpp::complex<double>> result;
    EXPECT_NO_THROW(
        { result = mgcpp::strict::cfft(vec, mgcpp::fft_direction::inverse); });

    EXPECT_EQ(result.size(), size);
    for (auto i = 0u; i < result.size(); ++i) {
      EXPECT_NEAR(result.check_value(i).real(), expected[i].real(), 1e-5)
          << "size = " << size << ", i = " << i;
      EXPECT_NEAR(result.check_value(i).imag(), expected[i].imag(), 1e-5)
          << "size = " << size << ", i = " << i;
    }
  }
}

TEST(fft_operation, float_complex_roundtrip_2d) {
  auto shape = mgcpp::make_shape(32, 16);
  mgcpp::device_matrix<mgcpp::complex<float>> mat(shape);
  for (auto i = 0u; i < shape[0]; ++i)
    for (auto j = 0u; j < shape[1]; ++j)
      mat.set_value(i, j, std::complex<float>(dist(rng), dist(rng)));

  mgcpp::device_matrix<mgcpp::complex<float>> F;
  EXPECT_NO_THROW(
      { F = mgcpp::strict::cfft(mat, mgcpp::fft_direction::forward); });

  mgcpp::device_matrix<mgcpp::complex<float>> result;
  EXPECT_NO_THROW(
      { result = mgcpp::strict::cfft(F, mgcpp::fft_direction::inverse); });

  EXPECT_EQ(result.shape(), mat.shape());
  for (auto i = 0u; i < result.shape()[0]; ++i) {
    for (auto j = 0u; j < result.shape()[1]; ++j) {
      EXPECT_NEAR(result.check_value(i, j).real(), mat.check_value(i, j).real(),
                  1e-3)
          << "size = (" << shape[0] << ", " << shape[1] << "), i = " << i
          << ", j = " << j;
      EXPECT_NEAR(result.check_value(i, j).imag(), mat.check_value(i, j).imag(),
                  1e-3)
          << "size = (" << shape[0] << ", " << shape[1] << "), i = " << i
          << ", j = " << j;
    }
  }
}
