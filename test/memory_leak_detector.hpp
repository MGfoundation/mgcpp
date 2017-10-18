
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TEST_MEMORY_LEAK_DETECTOR_HPP_
#define _MGCPP_TEST_MEMORY_LEAK_DETECTOR_HPP_

#include <gtest/gtest.h>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    class memory_leak_detector
        : public ::testing::EmptyTestEventListener
    {
        size_t free_memory;

        void OnTestStart(
            const ::testing::TestInfo& test_info) override
        {
            (void)test_info;

            auto memstat = cuda_mem_get_info();
            EXPECT_NO_THROW({
                    if(!memstat)
                        MGCPP_THROW_SYSTEM_ERROR(
                            memstat.error());
                });

            free_memory = memstat.value().first;
        }

        void OnTestEnd(
            const ::testing::TestInfo& test_info) override
        {
            auto memstat = cuda_mem_get_info();

            EXPECT_NO_THROW({
                    if(!memstat)
                        MGCPP_THROW_SYSTEM_ERROR(
                            memstat.error());
                });

            size_t result_memory = memstat.value().first;
            EXPECT_EQ(result_memory, free_memory)
                << "[ memory leak ] detected in test "
                << test_info.name();
        }
    };
}

#endif
