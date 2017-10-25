
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/operations/sum.hpp>
#include <mgcpp/operations/add.hpp>
#include <mgcpp/gpu/vector.hpp>

TEST(vec_vec_operation, vec_sum)
{
    size_t size = 10;
    float init_val = 3;
    mgcpp::gpu::vector<float> vec(size, init_val);

    float result = 0;
    EXPECT_NO_THROW({result = mgcpp::strict::sum(vec);});

    EXPECT_EQ(result, static_cast<float>(size) * init_val);
}

TEST(vec_vec_operation, vec_add)
{
    size_t size = 5;
    float first_init_val = 3;
    mgcpp::gpu::vector<float> first(size, first_init_val);

    float second_init_val = 4;
    mgcpp::gpu::vector<float> second(size, second_init_val);

    mgcpp::gpu::vector<float> result{};
    EXPECT_NO_THROW({result = mgcpp::strict::add(first, second);});

    for(auto i = 0u; i < size; ++i)
    {
        EXPECT_EQ(result.check_value(i),
                  first_init_val + second_init_val);
    }
}
