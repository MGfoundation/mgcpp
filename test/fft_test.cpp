
#include <gtest/gtest.h>

#include <mgcpp/operations/fft.hpp>

TEST(fft_operation, float_real_to_complex_fwd_fft)
{
    mgcpp::device_vector<float> vec({
        1, 2, 1, -1, 1, 1, 1, 3, 1, 3, 1, 3, 1, 2, 1, 3
    });

    size_t size = vec.size();

    mgcpp::device_vector<float> result;
    EXPECT_NO_THROW({result = mgcpp::strict::rfft(vec);});

    float expected[] = {
        24.000000, 0.000000,
        -2.071930, 5.002081,
        4.242640, 1.414214,
        2.388955, -0.989537,
        0.000000, 0.000000,
        -2.388955, -0.989537,
        -4.242640, 1.414214,
        2.071930, 5.002081,
        -8.000000, 0.000000,
    };

    EXPECT_EQ(result.size(), size / 2 * 2 + 2);
    for (auto i = 0u; i < result.size(); ++i) {
        EXPECT_NEAR(result.check_value(i), expected[i], 1e-5);
    }
}

#include <mgcpp/kernels/bits/fft.cuh>
TEST(fft_operation, float_real_to_complex_fwd_fft_custom_kernel)
{
    mgcpp::device_vector<float> vec({
        1, 2, 1, -1, 1, 1, 1, 3, 1, 3, 1, 3, 1, 2, 1, 3
    });

    size_t size = vec.size();

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

    float expected[] = {
        24.000000, 0.000000,
        -2.071930, 5.002081,
        4.242640, 1.414214,
        2.388955, -0.989537,
        0.000000, 0.000000,
        -2.388955, -0.989537,
        -4.242640, 1.414214,
        2.071930, 5.002081,
        -8.000000, 0.000000,
        2.07193, -5.00208,
        -4.24264, -1.41421,
        -2.38896, 0.989538,
        0., 0.,
        2.38896, 0.989538,
        4.24264, -1.41421,
        -2.07193, -5.00208
    };

    EXPECT_EQ(result.size(), size * 2);
    for (auto i = 0u; i < result.size(); ++i) {
        EXPECT_NEAR(result.check_value(i), expected[i], 1e-5);
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
