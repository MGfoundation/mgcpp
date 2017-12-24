
#include <gtest/gtest.h>

#include <mgcpp/operations/fft.hpp>

TEST(fft_operation, float_real_to_complex_fwd_fft)
{
    mgcpp::device_vector<float> vec({
        1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3
    });

    size_t size = vec.size();

    mgcpp::device_vector<float> result;
    EXPECT_NO_THROW({result = mgcpp::rfft(vec);});

    float expected[] = {
        32.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,
        0.f, 0.f,

        -16.f, 0.f
    };

    EXPECT_EQ(result.size(), size / 2 * 2 + 2);
    for (auto i = 0u; i < size; ++i) {
        EXPECT_EQ(result.check_value(i), expected[i]);
    }
}

TEST(fft_operation, double_real_to_complex_fwd_fft)
{
    mgcpp::device_vector<double> vec({
        1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3
    });

    size_t size = vec.size();

    mgcpp::device_vector<double> result;
    EXPECT_NO_THROW({result = mgcpp::rfft(vec);});

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
    for (auto i = 0u; i < size; ++i) {
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
    EXPECT_NO_THROW({result = mgcpp::irfft(vec);});

    const size_t size = 16;
    float expected[size] = {
        1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3
    };

    EXPECT_EQ(result.size(), size);
    for (auto i = 0u; i < size; ++i) {
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
    EXPECT_NO_THROW({result = mgcpp::irfft(vec);});

    const size_t size = 16;
    double expected[size] = {
        1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3
    };

    EXPECT_EQ(result.size(), size);
    for (auto i = 0u; i < size; ++i) {
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
    EXPECT_NO_THROW({result = mgcpp::cfft(vec, mgcpp::fft_direction::forward);});

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
    for (auto i = 0u; i < size * 2; ++i) {
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
    EXPECT_NO_THROW({result = mgcpp::cfft(vec, mgcpp::fft_direction::inverse);});

    float expected[] = {
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0
    };

    size_t size = 16;

    EXPECT_EQ(result.size(), size * 2);
    for (auto i = 0u; i < size * 2; ++i) {
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
    EXPECT_NO_THROW({result = mgcpp::cfft(vec, mgcpp::fft_direction::forward);});

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
    for (auto i = 0u; i < size * 2; ++i) {
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
    EXPECT_NO_THROW({result = mgcpp::cfft(vec, mgcpp::fft_direction::inverse);});

    double expected[] = {
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0,
        1, 0, 3, 0, 1, 0, 3, 0
    };

    size_t size = 16;

    EXPECT_EQ(result.size(), size * 2);
    for (auto i = 0u; i < size * 2; ++i) {
        EXPECT_EQ(result.check_value(i), expected[i]);
    }
}
