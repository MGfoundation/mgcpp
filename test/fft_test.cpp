
#include <gtest/gtest.h>

#include <mgcpp/operations/fft.hpp>

#define TEST_SIZE 2048

#include <random>
#include <valarray>
#include <complex>
using carray = std::valarray<std::complex<double>>;
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
        auto w = std::polar(1., x);
        for (int j = 0; j < n; j += i << 1) {
            std::complex<double> th = {1, 0};
            for (int k = 0; k < i; k++) {
                auto tmp = a[i + j + k] * th;
                a[i + j + k] = a[j + k] - tmp;
                a[j + k] += tmp;
                th *= w;
            }
        }
    }
    if (inv) {
        a /= std::complex<double>(n);
    }
}
std::default_random_engine rng;
std::uniform_real_distribution<double> dist(0.0, 1.0);

// rfft
TEST(fft_operation, float_real_to_complex_fwd_fft)
{
    for (size_t size = 1; size <= TEST_SIZE; size *= 2)
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
            EXPECT_NEAR(result.check_value(i).real(), expected[i].real(), 1e-4)
                << "size = " << size << ", i = " << i;
            EXPECT_NEAR(result.check_value(i).imag(), expected[i].imag(), 1e-4)
                << "size = " << size << ", i = " << i;
        }
    }
}

TEST(fft_operation, double_real_to_complex_fwd_fft)
{
    for (size_t size = 1; size <= TEST_SIZE; size *= 2)
    {
        mgcpp::device_vector<double> vec(size);
        for (auto i = 0u; i < size; ++i) vec.set_value(i, dist(rng));

        carray expected(size);
        for (auto i = 0u; i < vec.size(); ++i)
            expected[i] = {vec.check_value(i), 0};
        fft(expected, false);

        mgcpp::device_vector<mgcpp::complex<double>> result;
        EXPECT_NO_THROW({ result = mgcpp::strict::rfft(vec); });

        EXPECT_EQ(result.size(), size / 2 + 1);
        for (auto i = 0u; i < result.size(); ++i) {
            EXPECT_NEAR(result.check_value(i).real(), expected[i].real(), 1e-4)
                << "size = " << size << ", i = " << i;
            EXPECT_NEAR(result.check_value(i).imag(), expected[i].imag(), 1e-4)
                << "size = " << size << ", i = " << i;
        }
    }
}

// irfft
TEST(fft_operation, float_complex_to_real_inv_fft)
{
    for (size_t size = 2; size <= TEST_SIZE; size *= 2)
    {
        mgcpp::device_vector<mgcpp::complex<float>> vec(size / 2 + 1);
        for (auto i = 0u; i < vec.size(); ++i) {
            std::complex<float> random_complex(dist(rng), dist(rng));
            vec.set_value(i, random_complex);
        }

        carray expected(size);
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
            EXPECT_NEAR(result.check_value(i), expected[i].real(), 1e-4)
                << "size = " << size << ", i = " << i;
        }
    }
}

TEST(fft_operation, double_complex_to_real_inv_fft)
{
    for (size_t size = 2; size <= TEST_SIZE; size *= 2)
    {
        mgcpp::device_vector<mgcpp::complex<double>> vec(size / 2 + 1);
        for (auto i = 0u; i < vec.size(); ++i) {
            std::complex<double> random_complex(dist(rng), dist(rng));
            vec.set_value(i, random_complex);
        }

        carray expected(size);
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
            EXPECT_NEAR(result.check_value(i), expected[i].real(), 1e-4)
                << "size = " << size << ", i = " << i;
        }
    }
}

// forward cfft
TEST(fft_operation, float_complex_to_complex_fwd_fft)
{
    for (size_t size = 1; size <= TEST_SIZE; size *= 2)
    {
        mgcpp::device_vector<mgcpp::complex<float>> vec(size);
        for (auto i = 0u; i < size; ++i) {
            std::complex<float> random_complex(dist(rng), dist(rng));
            vec.set_value(i, random_complex);
        }

        carray expected(size);
        for (auto i = 0u; i < vec.size(); ++i) {
            expected[i] = vec.check_value(i);
        }
        fft(expected, false);

        mgcpp::device_vector<mgcpp::complex<float>> result;
        EXPECT_NO_THROW({ result = mgcpp::strict::cfft(vec, mgcpp::fft_direction::forward); });

        EXPECT_EQ(result.size(), size);
        for (auto i = 0u; i < result.size(); ++i) {
            EXPECT_NEAR(result.check_value(i).real(), expected[i].real(), 1e-4)
                << "size = " << size << ", i = " << i;
            EXPECT_NEAR(result.check_value(i).imag(), expected[i].imag(), 1e-4)
                << "size = " << size << ", i = " << i;
        }
    }
}

TEST(fft_operation, double_complex_to_complex_fwd_fft)
{
    for (size_t size = 1; size <= TEST_SIZE; size *= 2)
    {
        mgcpp::device_vector<mgcpp::complex<double>> vec(size);
        for (auto i = 0u; i < size; ++i) {
            std::complex<double> random_complex(dist(rng), dist(rng));
            vec.set_value(i, random_complex);
        }

        carray expected(size);
        for (auto i = 0u; i < vec.size(); ++i) {
            expected[i] = vec.check_value(i);
        }
        fft(expected, false);

        mgcpp::device_vector<mgcpp::complex<double>> result;
        EXPECT_NO_THROW({ result = mgcpp::strict::cfft(vec, mgcpp::fft_direction::forward); });

        EXPECT_EQ(result.size(), size);
        for (auto i = 0u; i < result.size(); ++i) {
            EXPECT_NEAR(result.check_value(i).real(), expected[i].real(), 1e-4)
                << "size = " << size << ", i = " << i;
            EXPECT_NEAR(result.check_value(i).imag(), expected[i].imag(), 1e-4)
                << "size = " << size << ", i = " << i;
        }
    }
}

// inverse cfft
TEST(fft_operation, float_complex_to_complex_inv_fft)
{
    for (size_t size = 1; size <= TEST_SIZE; size *= 2)
    {
        mgcpp::device_vector<mgcpp::complex<float>> vec(size);
        for (auto i = 0u; i < size; ++i) {
            std::complex<float> random_complex(dist(rng), dist(rng));
            vec.set_value(i, random_complex);
        }

        carray expected(size);
        for (auto i = 0u; i < vec.size(); ++i) {
            expected[i] = vec.check_value(i);
        }
        fft(expected, true);

        mgcpp::device_vector<mgcpp::complex<float>> result;
        EXPECT_NO_THROW({ result = mgcpp::strict::cfft(vec, mgcpp::fft_direction::inverse); });

        EXPECT_EQ(result.size(), size);
        for (auto i = 0u; i < result.size(); ++i) {
            EXPECT_NEAR(result.check_value(i).real(), expected[i].real(), 1e-4)
                << "size = " << size << ", i = " << i;
            EXPECT_NEAR(result.check_value(i).imag(), expected[i].imag(), 1e-4)
                << "size = " << size << ", i = " << i;
        }
    }
}

TEST(fft_operation, double_complex_to_complex_inv_fft)
{
    for (size_t size = 1; size <= TEST_SIZE; size *= 2)
    {
        mgcpp::device_vector<mgcpp::complex<double>> vec(size);
        for (auto i = 0u; i < size; ++i) {
            std::complex<double> random_complex(dist(rng), dist(rng));
            vec.set_value(i, random_complex);
        }

        carray expected(size);
        for (auto i = 0u; i < vec.size(); ++i) {
            expected[i] = vec.check_value(i);
        }
        fft(expected, true);

        mgcpp::device_vector<mgcpp::complex<double>> result;
        EXPECT_NO_THROW({ result = mgcpp::strict::cfft(vec, mgcpp::fft_direction::inverse); });

        EXPECT_EQ(result.size(), size);
        for (auto i = 0u; i < result.size(); ++i) {
            EXPECT_NEAR(result.check_value(i).real(), expected[i].real(), 1e-4)
                << "size = " << size << ", i = " << i;
            EXPECT_NEAR(result.check_value(i).imag(), expected[i].imag(), 1e-4)
                << "size = " << size << ", i = " << i;
        }
    }
}
