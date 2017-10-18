
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>
#include <mgcpp/cuda/device.hpp>

#include "memory_leak_detector.hpp"

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::UnitTest::GetInstance()
          ->listeners()
          .Append(new mgcpp::memory_leak_detector());
    return RUN_ALL_TESTS();
}
