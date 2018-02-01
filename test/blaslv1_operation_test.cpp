
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/vector/device_vector.hpp>
#include <mgcpp/operations/abs.hpp>
#include <mgcpp/operations/add.hpp>
#include <mgcpp/operations/hdmd.hpp>
#include <mgcpp/operations/mean.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/operations/sub.hpp>
#include <mgcpp/operations/sum.hpp>

TEST(vec_vec_operation, vec_sum)
{
    size_t size = 10;
    float init_val = 3;
    mgcpp::device_vector<float> vec(size, init_val);

    float result = 0;
    EXPECT_NO_THROW({result = mgcpp::strict::sum(vec);});

    EXPECT_EQ(result, static_cast<float>(size) * init_val);
}

TEST(vec_vec_operation, vec_add)
{
    size_t size = 5;
    float first_init_val = 3;
    mgcpp::device_vector<float> first(size, first_init_val);

    float second_init_val = 4;
    mgcpp::device_vector<float> second(size, second_init_val);

    mgcpp::device_vector<float> result{};
    EXPECT_NO_THROW({result = mgcpp::strict::add(first, second);});

    for(auto i = 0u; i < size; ++i)
    {
        EXPECT_EQ(result.check_value(i),
                  first_init_val + second_init_val);
    }
}

TEST(vec_vec_operation, vec_sub)
{
    size_t size = 5;
    float first_init_val = 4;
    mgcpp::device_vector<float> first(size, first_init_val);

    float second_init_val = 3;
    mgcpp::device_vector<float> second(size, second_init_val);

    mgcpp::device_vector<float> result{};
    EXPECT_NO_THROW({result = mgcpp::strict::sub(first, second);});

    for(auto i = 0u; i < size; ++i)
    {
        EXPECT_EQ(result.check_value(i),
                  first_init_val - second_init_val);
    }
}

TEST(vec_operation, vec_scalar_mult)
{
    size_t size = 5;
    float init_val = 3;
    mgcpp::device_vector<float> vec(size, init_val);

    float scalar = 4;

    mgcpp::device_vector<float> result{};
    EXPECT_NO_THROW({
            result = mgcpp::strict::mult(scalar, vec);
        });

    for(auto i = 0u; i < size; ++i)
    {
        EXPECT_EQ(result.check_value(i), init_val * scalar);
    }
}

TEST(vec_operation, vec_abs)
{
    mgcpp::device_vector<float> vec{-1, -2, -3};

    mgcpp::device_vector<float> result{};
    EXPECT_NO_THROW({
            result = mgcpp::strict::abs(vec);

            EXPECT_EQ(result.check_value(0), 1);
            EXPECT_EQ(result.check_value(1), 2);
            EXPECT_EQ(result.check_value(2), 3);
        });
}

TEST(vec_operation, vec_mean)
{
    mgcpp::device_vector<float> vec{1, 2, 3};

    EXPECT_NO_THROW({
            float result = mgcpp::strict::mean(vec);

            EXPECT_EQ(result, 2);
        });
}

TEST(vec_vec_operation, vec_hadamard_product)
{
    size_t size = 5;
    float first_val = 3;
    float second_val = 5;
    mgcpp::device_vector<float> first(size, first_val);
    mgcpp::device_vector<float> second(size, second_val);

    mgcpp::device_vector<float> result{};
    EXPECT_NO_THROW({
            result = mgcpp::strict::hdmd(first, second);
        });

    for(auto i = 0u; i < size; ++i)
    {
        EXPECT_EQ(result.check_value(i), first_val * second_val);
    }
}
