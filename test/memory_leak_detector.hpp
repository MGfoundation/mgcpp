
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TEST_MEMORY_LEAK_DETECTOR_HPP_
#define _MGCPP_TEST_MEMORY_LEAK_DETECTOR_HPP_

#include <vector>

#include <gtest/gtest.h>

#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    class memory_leak_detector
        : public ::testing::EmptyTestEventListener
    {
        std::vector<size_t> device_free_memory;

        void OnTestStart(
            ::testing::TestInfo const& test_info) override;

        void OnTestEnd(
            ::testing::TestInfo const& test_info) override;
    };
}

#endif
