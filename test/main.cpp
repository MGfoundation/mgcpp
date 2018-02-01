
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/global/init.hpp>

#include "memory_leak_detector.hpp"

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    ::testing::UnitTest::GetInstance()
          ->listeners()
          .Append(new mgcpp::memory_leak_detector());

    mgcpp::init(true);

    return RUN_ALL_TESTS();
}
