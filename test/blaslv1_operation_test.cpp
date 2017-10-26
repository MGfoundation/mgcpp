
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/operations/sum.hpp>
#include <mgcpp/operations/add.hpp>
#include <mgcpp/operations/sub.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/gpu/vector.hpp>

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
