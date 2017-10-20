
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>

#include "memory_leak_detector.hpp"

namespace mgcpp
{
    void
    memory_leak_detector::
    OnTestStart(::testing::TestInfo const& test_info) 
    {
        (void)test_info;

        std::error_code status = cudaGetDeviceCount(&device_number);
        if(status != status_t::success)
            MGCPP_THROW_SYSTEM_ERROR(status);

        device_free_memory.clear();
        device_free_memory.reserve(device_number);

        for(int i = 0; i < device_number; ++i)
        {
            (void)cuda_set_device(static_cast<size_t>(i));

            auto memstat = cuda_mem_get_info();
            EXPECT_TRUE(memstat)
                << "error occurred while getting memory of device "
                << i << '\n'
                << memstat.error()
                << "in start of test "
                << test_info.name();

            device_free_memory.push_back(memstat.value().first);
        }
    }

    void
    memory_leak_detector::
    OnTestEnd(::testing::TestInfo const& test_info) 
    {
        for(int i = 0; i < device_number; ++i)
        {
            (void)cuda_set_device(static_cast<size_t>(i));

            auto memstat = cuda_mem_get_info();
            
            EXPECT_TRUE(memstat)
                << "error while getting memory info of device "
                << i << '\n'
                << "in test "
                << test_info.name();

            size_t result_memory = memstat.value().first;
            EXPECT_EQ(result_memory, device_free_memory[i])
                << "[ memory leak ] detected in test "
                << test_info.name() << '\n'
                << "for device "
                << i << '\n'
                << "in test "
                << test_info.name();
        }
    }
}
