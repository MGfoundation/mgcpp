

//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <thread>

#include <mgcpp/context/global_context.hpp>

#include <gtest/gtest.h>

TEST(global_context, request_context_from_same_thread)
{
    auto* context_one = &mgcpp::global_context::get_thread_context();
    auto* context_two = &mgcpp::global_context::get_thread_context();

    EXPECT_EQ(context_one, context_two);

    mgcpp::global_context::reference_cnt_decr();
    mgcpp::global_context::reference_cnt_decr();
}

TEST(global_context, request_context_from_different_thread)
{
    auto* context_one = &mgcpp::global_context::get_thread_context();

    mgcpp::thread_context* context_two;
    auto thread = std::thread(
        [&context_two, &context_one]()
        {
            context_two =
            &mgcpp::global_context::get_thread_context();

            EXPECT_NE(context_one, context_two);

            mgcpp::global_context::reference_cnt_decr();
        });

    thread.join();
    mgcpp::global_context::reference_cnt_decr();
}
