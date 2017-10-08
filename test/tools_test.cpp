
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

    EXPECT_TRUE(checker);
}

TEST(leak_check, case_no_leak)
{
    auto checker = mgcpp::leak_checker();

    EXPECT_TRUE(checker.cache());
}

TEST(leak_check, case_no_leak_cached_result)
{
    auto checker = mgcpp::leak_checker();

    checker.cache();
    EXPECT_TRUE(checker);
}

TEST(leak_check, case_leak_bool_operator)
{
    auto checker = mgcpp::leak_checker(); 

    auto mem = mgcpp::cuda_malloc<float>(10);
    EXPECT_TRUE(mem);

    EXPECT_FALSE(checker);

    (void)mgcpp::cuda_free(mem.value());
}

TEST(leak_check, case_leak)
{
    auto checker = mgcpp::leak_checker(); 

    auto mem = mgcpp::cuda_malloc<float>(10);
    EXPECT_TRUE(mem);

    EXPECT_FALSE(checker.cache());

    (void)mgcpp::cuda_free(mem.value());
}

TEST(leak_check, case_leak_cached_result)
{
    auto checker = mgcpp::leak_checker(); 

    auto mem = mgcpp::cuda_malloc<float>(10);
    EXPECT_TRUE(mem);

    checker.cache();
    EXPECT_FALSE(static_cast<bool>(checker));

    (void)mgcpp::cuda_free(mem.value());
}
