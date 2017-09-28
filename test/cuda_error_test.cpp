
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)


#include <gtest/gtest.h>

#include <mgcpp/system/cuda_error.hpp>

#include <string>
#include <system_error>

TEST(cuda_error, cuda_error_string_function)
{
    using mgcpp::cuda_error_t;
    std::error_code err_code =
        cuda_error_t::cudaErrorMemoryAllocation;
    auto result = std::string(err_code.message());
    auto answer = std::string("out of memory");

    ASSERT_EQ(result, answer);
}

