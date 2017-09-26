
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <string>

#include <gtest/gtest.h>

#include <mgcpp/cuda/internal/cuda_error.hpp>

TEST(cuda_error, cuda_error_string_function)
{
    using mgcpp::internal::cuda_error_t;
    using mgcpp::internal::cuda_error_string;

    cuda_error_t err_code = cuda_error_t::memory_allocation;
    auto result = std::string(cuda_error_string(err_code));
    auto answer = std::string("out of memory");

    ASSERT_EQ(result, answer);
}

