
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <thread>

#include <mgcpp/context/global_context.hpp>
#include "test_policy.hpp"

#include <gtest/gtest.h>

TEST(thead_context, request_cublas_handle)
{
    auto* context = &mgcpp::global_context::get_thread_context();

    auto cublas_context_one = context->get_cublas_context(0);
    auto cublas_context_two = context->get_cublas_context(0);

    EXPECT_EQ(cublas_context_one, cublas_context_two);

    mgcpp::global_context::reference_cnt_decr();
}

TEST(thead_context, request_cublas_handle_different_device)
{
    if(mgcpp::test_policy::get_policy().device_num() >= 2)
    {
        auto* context = &mgcpp::global_context::get_thread_context();

    auto cublas_context_one = context->get_cublas_context(0);
    auto cublas_context_two = context->get_cublas_context(1);

    EXPECT_NE(cublas_context_one, cublas_context_two);

    mgcpp::global_context::reference_cnt_decr();
    }
}
