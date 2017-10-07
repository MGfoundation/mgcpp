
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>

#include <mgcpp/tools/memory_check.hpp>
#include <mgcpp/cuda/memory.hpp>

TEST(leak_check, case_no_leak_bool_operator)
{
    auto checker = mgcpp::leak_checker();

    EXPECT_TRUE(static_cast<bool>(checker));
}

TEST(leak_check, case_no_leak)
{
    auto checker = mgcpp::leak_checker();

    EXPECT_TRUE(checker.check());
}

TEST(leak_check, case_no_leak_cached_result)
{
    auto checker = mgcpp::leak_checker();

    checker.check();
    EXPECT_TRUE(static_cast<bool>(checker));
}

TEST(leak_check, case_leak_bool_operator)
{
    auto checker = mgcpp::leak_checker(); 

    auto mem = mgcpp::cuda_malloc<float>(10);
    EXPECT_TRUE(mem);

    EXPECT_FALSE(static_cast<bool>(checker));

    (void)mgcpp::cuda_free(mem.value());
}

TEST(leak_check, case_leak)
{
    auto checker = mgcpp::leak_checker(); 

    auto mem = mgcpp::cuda_malloc<float>(10);
    EXPECT_TRUE(mem);

    EXPECT_FALSE(checker.check());

    (void)mgcpp::cuda_free(mem.value());
}

TEST(leak_check, case_leak_cached_result)
{
    auto checker = mgcpp::leak_checker(); 

    auto mem = mgcpp::cuda_malloc<float>(10);
    EXPECT_TRUE(mem);

    checker.check();
    EXPECT_FALSE(static_cast<bool>(checker));

    (void)mgcpp::cuda_free(mem.value());
}
