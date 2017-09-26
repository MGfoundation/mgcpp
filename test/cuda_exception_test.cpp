
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/cuda/internal/status_wrapper.hpp>
#include <mgcpp/cuda/stdlib.hpp>
#include <mgcpp/cuda/exception.hpp>

TEST(mgcpp_exception, mgcpp_error_check)
{
    using mgcpp::internal::cuda_mem_get_info;

    size_t free_memory = 0;
    cuda_mem_get_info(&free_memory, nullptr);

    float* ptr = nullptr;
    (void)ptr; // warning suppression

    EXPECT_EXIT({
            mgcpp_error_check(
                ptr = mgcpp::cuda_malloc<float>(free_memory * 2));
        }, ::testing::ExitedWithCode(1), "");
}
