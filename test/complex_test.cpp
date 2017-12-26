
#include <gtest/gtest.h>

#include <mgcpp/global/complex.hpp>

TEST(complex, complex) {
    mgcpp::complex<float> c;
    EXPECT_EQ(c.real, 0);
    EXPECT_EQ(c.imag, 0);

    c = {1, 2};
    EXPECT_EQ(c.real, 1);
    EXPECT_EQ(c.imag, 2);

    c *= {2, 3};
    EXPECT_EQ(c.real, -4);
    EXPECT_EQ(c.imag, 7);

    c -= {-4, 7};
    EXPECT_EQ(c.real, 0);
    EXPECT_EQ(c.imag, 0);

    c = {3, 4};
    c /= 2;
    EXPECT_EQ(c.real, 1.5);
    EXPECT_EQ(c.imag, 2);
}
