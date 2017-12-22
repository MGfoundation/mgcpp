
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
