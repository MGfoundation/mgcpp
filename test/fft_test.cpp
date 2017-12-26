
#include <gtest/gtest.h>

#include <mgcpp/operations/fft.hpp>

#include <random>
#include <complex>
#include <valarray>
using complex = std::complex<double>;
using carray = std::valarray<complex>;
constexpr double PI = 3.1415926535897932384626433832795028;
void fft(carray &a, bool inv)
{
    int n = a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        complex wlen = std::polar(1., 2 * PI / len * (inv? -1: 1));
        for (int i = 0; i < n; i += len) {
            complex w(1);
            for (int j = 0; j < len / 2; j++){
                complex u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inv) {
        for (int i = 0; i < n; i++) a[i] /= n;
    }
}

std::default_random_engine rng;
std::uniform_real_distribution<double> dist(0.0, 1.0);

TEST(fft_operation, float_real_to_complex_fwd_fft)
{
    size_t size = 1024;

    mgcpp::device_vector<float> vec(size);
    for (auto i = 0u; i < size; ++i) vec.set_value(i, dist(rng));

    carray expected(size);
    for (auto i = 0u; i < vec.size(); ++i)
        expected[i] = vec.check_value(i);
    fft(expected, false);

    mgcpp::device_vector<float> result;
    EXPECT_NO_THROW({result = mgcpp::strict::rfft(vec);});

    EXPECT_EQ(result.size(), size / 2 * 2 + 2);
    for (auto i = 0u; i < result.size() / 2; ++i) {
        EXPECT_NEAR(result.check_value(i * 2), expected[i].real(), 1e-4);
    }
}

#include <mgcpp/kernels/bits/fft.cuh>
TEST(fft_operation, float_real_to_complex_fwd_fft_custom_kernel)
{
    size_t size = 1024;

    mgcpp::device_vector<float> vec(size);
    for (auto i = 0u; i < size; ++i) vec.set_value(i, dist(rng));

    carray expected(size);
    for (auto i = 0u; i < vec.size(); ++i)
        expected[i] = vec.check_value(i);
    fft(expected, false);

    mgcpp::device_vector<float> result(size * 2);

    for (auto i = 1u, j = 0u; i < size; i++) {
        auto b = size >> 1;
        for (; j >= b; b >>= 1) j -= b;
        j += b;
        if (i < j) {
            float t = vec.check_value(i);
            vec.set_value(i, vec.check_value(j));
            vec.set_value(j, t);
        }
    }
    mgcpp::mgblas_Srfft(vec.data(), result.data_mutable(), size);

    //EXPECT_EQ(result.size(), size * 2);

    for (auto i = 0u; i < result.size() / 2; ++i) {
        EXPECT_NEAR(result.check_value(i * 2), expected[i].real(), 1e-4);
    }
}

TEST(fft_operation, double_real_to_complex_fwd_fft)
{
    mgcpp::device_vector<double> vec({
        1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3
    });

    size_t size = vec.size();

    mgcpp::device_vector<double> result;
    EXPECT_NO_THROW({result = mgcpp::strict::rfft(vec);});

    double expected[] = {
        32., 0.,
        0., 0.,
        0., 0.,
        0., 0.,

        0., 0.,
        0., 0.,
        0., 0.,
        0., 0.,

        -16., 0.
    };

    EXPECT_EQ(result.size(), size / 2 * 2 + 2);
    for (auto i = 0u; i < result.size(); ++i) {
        EXPECT_EQ(result.check_value(i), expected[i]);
    }
}

TEST(fft_operation, float_complex_to_real_inv_fft)
{
    mgcpp::device_vector<float> vec({
        32.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        -16.f, 0.f
    });

    mgcpp::device_vector<float> result;
    EXPECT_NO_THROW({result = mgcpp::strict::irfft(vec);});

    const size_t size = 16;
    float expected[size] = {
        1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3
    };

    EXPECT_EQ(result.size(), size);
    for (auto i = 0u; i < result.size(); ++i) {
        EXPECT_EQ(result.check_value(i), expected[i]);
    }
}

TEST(fft_operation, double_complex_to_real_inv_fft)
{
    mgcpp::device_vector<double> vec({
        32.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        -16.f, 0.f
    });

    mgcpp::device_vector<double> result;
    EXPECT_NO_THROW({result = mgcpp::strict::irfft(vec);});

    const size_t size = 16;
    double expected[size] = {
        1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3
    };

    EXPECT_EQ(result.size(), size);
    for (auto i = 0u; i < result.size(); ++i) {
        EXPECT_EQ(result.check_value(i), expected[i]);
    }
}

TEST(fft_operation, float_complex_to_complex_fwd_fft)
{
    mgcpp::device_vector<float> vec({
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0
    });

    mgcpp::device_vector<float> result;
    EXPECT_NO_THROW({result = mgcpp::strict::cfft(vec, mgcpp::fft_direction::forward);});

    float expected[] = {
        32.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        -16.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
    };

    size_t size = 16;

    EXPECT_EQ(result.size(), size * 2);
    for (auto i = 0u; i < result.size(); ++i) {
        EXPECT_EQ(result.check_value(i), expected[i]);
    }
}

TEST(fft_operation, float_complex_to_complex_inv_fft)
{
    mgcpp::device_vector<float> vec({
        32.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        -16.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
    });

    mgcpp::device_vector<float> result;
    EXPECT_NO_THROW({result = mgcpp::strict::cfft(vec, mgcpp::fft_direction::inverse);});

    float expected[] = {
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0
    };

    size_t size = 16;

    EXPECT_EQ(result.size(), size * 2);
    for (auto i = 0u; i < result.size(); ++i) {
        EXPECT_EQ(result.check_value(i), expected[i]);
    }
}

TEST(fft_operation, double_complex_to_complex_fwd_fft)
{
    mgcpp::device_vector<double> vec({
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0
    });

    mgcpp::device_vector<double> result;
    EXPECT_NO_THROW({result = mgcpp::strict::cfft(vec, mgcpp::fft_direction::forward);});

    double expected[] = {
        32.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        -16.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
    };

    size_t size = 16;

    EXPECT_EQ(result.size(), size * 2);
    for (auto i = 0u; i < result.size(); ++i) {
        EXPECT_EQ(result.check_value(i), expected[i]);
    }
}

TEST(fft_operation, double_complex_to_complex_inv_fft)
{
    mgcpp::device_vector<double> vec({
        32.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        -16.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
    });

    mgcpp::device_vector<double> result;
    EXPECT_NO_THROW({result = mgcpp::strict::cfft(vec, mgcpp::fft_direction::inverse);});

    double expected[] = {
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0
    };

    size_t size = 16;

    EXPECT_EQ(result.size(), size * 2);
    for (auto i = 0u; i < result.size(); ++i) {
        EXPECT_EQ(result.check_value(i), expected[i]);
    }
}
