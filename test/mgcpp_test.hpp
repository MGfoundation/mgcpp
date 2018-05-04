
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TEST_TEST_FIXTURE_HPP_
#define _MGCPP_TEST_TEST_FIXTURE_HPP_

#include <gtest/gtest.h>

#define MGCPP_TEST(FIXTURE, TEST_NAME, TEST_CASE) \
    TEST_F(FIXTURE, TEST_NAME){TEST_CASE} 

#endif
