
#include <gtest/gtest.h>

#include <mgcpp/operations/fft.hpp>

TEST(fft_operation, float_real_to_complex_fwd_fft)
{
    size_t size = 10;
    float init_val = 3;
    mgcpp::device_vector<float> vec(size, init_val);

    mgcpp::device_vector<float> result;
    EXPECT_NO_THROW({result = mgcpp::fft(result);});

    //EXPECT_EQ(result, /* something */);
}
