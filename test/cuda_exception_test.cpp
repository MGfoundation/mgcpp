
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/system/exception.hpp>

TEST(mgcpp_exception, mgcpp_error_check)
{
    size_t free_memory = 0;
    cudaMemGetInfo(&free_memory, nullptr);

    EXPECT_EXIT(
        {
            mgcpp_error_check(
                auto rst = mgcpp::cuda_malloc<float>(free_memory * 2);
                if(!rst)
                    MGCPP_THROW_SYSTEM_ERROR(rst.error()));
        }, ::testing::ExitedWithCode(1), "");
}
