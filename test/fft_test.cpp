
#include <gtest/gtest.h>

#include <mgcpp/operations/fft.hpp>

#include <random>
#include <valarray>
using carray = std::valarray<mgcpp::complex<double>>;
constexpr double PI = 3.1415926535897932384626433832795028;
void fft(carray &a, bool inv)
{
    int n = a.size();
    for (int i = 1, j = 0; i < n; i++){
        int bit = n >> 1;
        while (!((j ^= bit) & bit)) bit >>= 1;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int i = 1; i < n; i <<= 1) {
        double x = inv ? PI / i : -PI / i;
        auto w = mgcpp::polar(1., x);
        for (int j = 0; j < n; j += i << 1) {
            mgcpp::complex<double> th = {1, 0};
            for (int k = 0; k < i; k++) {
                auto tmp = a[i + j + k] * th;
                a[i + j + k] = a[j + k] - tmp;
                a[j + k] += tmp;
                th *= w;
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
    for (size_t size = 1; size <= 2048; size *= 2)
    {
        mgcpp::device_vector<float> vec(size);
        for (auto i = 0u; i < size; ++i) vec.set_value(i, dist(rng));

        carray expected(size);
        for (auto i = 0u; i < vec.size(); ++i)
            expected[i] = {vec.check_value(i), 0};
        fft(expected, false);

        mgcpp::device_vector<mgcpp::complex<float>> result;
        EXPECT_NO_THROW({ result = mgcpp::strict::rfft(vec); });

        EXPECT_EQ(result.size(), size / 2 + 1);
        for (auto i = 0u; i < result.size(); ++i) {
            EXPECT_NEAR(result.check_value(i).real, expected[i].real, 1e-4);
            EXPECT_NEAR(result.check_value(i).imag, expected[i].imag, 1e-4);
        }
    }
}

#include <mgcpp/kernels/bits/fft.cuh>
TEST(fft_operation, float_real_to_complex_fwd_fft_custom_kernel)
{
    for (size_t size = 1; size <= 2048; size *= 2)
    {
        mgcpp::device_vector<float> vec(size);
        for (auto i = 0u; i < size; ++i) vec.set_value(i, dist(rng));

        carray expected(size);
        for (auto i = 0u; i < vec.size(); ++i)
            expected[i] = {vec.check_value(i), 0};
        fft(expected, false);

        mgcpp::device_vector<mgcpp::complex<float>> result(size);

        for (auto i = 1u, j = 0u; i < size; i++) {
            auto b = size >> 1;
            while (!((j ^= b) & b)) b >>= 1;
            if (i < j) {
                float t = vec.check_value(i);
                vec.set_value(i, vec.check_value(j));
                vec.set_value(j, t);
            }
        }
        mgcpp::mgblas_Srfft(vec.data(),
                            reinterpret_cast<float*>(result.data_mutable()),
                            size);

        EXPECT_EQ(result.size(), size / 2 + 1);
        for (auto i = 0u; i < result.size(); ++i) {
            EXPECT_NEAR(result.check_value(i).real, expected[i].real, 1e-4);
            EXPECT_NEAR(result.check_value(i).imag, expected[i].imag, 1e-4);
        }
    }
}

