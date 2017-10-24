
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/operations/sum.hpp>
#include <mgcpp/gpu/vector.hpp>

TEST(vec_vec_operation, vec_sum)
{
    size_t size = 10;
    float init_val = 3;
    mgcpp::gpu::vector<float> vec(size, init_val);


}
