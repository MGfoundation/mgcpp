
#include <thread>

#include <mgcpp/context/global_context.hpp>

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
    auto* context = &mgcpp::global_context::get_thread_context();

    auto cublas_context_one = context->get_cublas_context(0);
    auto cublas_context_two = context->get_cublas_context(1);

    EXPECT_NE(cublas_context_one, cublas_context_two);

    mgcpp::global_context::reference_cnt_decr();
}
