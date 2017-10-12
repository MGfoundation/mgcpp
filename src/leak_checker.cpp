
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/tools/memory_check.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/cuda/device.hpp>

namespace mgcpp
{
    leak_checker::
    leak_checker(size_t device_id) 
        : _device_id(device_id),
          _before_free_memory(0),
          _after_free_memory(0),
          _cached(false)
    {
        // auto set_result = cuda_set_device(_device_id);
        // if(!set_result)
        //     MGCPP_THROW_SYSTEM_ERROR(set_result.error());

        auto result = cuda_mem_get_info();
        if(!result)
            MGCPP_THROW_SYSTEM_ERROR(result.error());
        
        _before_free_memory = result.value().first;
    }

    bool
    leak_checker::
    cache() noexcept
    {
        // (void)cuda_set_device(_device_id);
        auto result = cuda_mem_get_info();

        _after_free_memory = result.value().first;
        _cached = true;

        return _after_free_memory == _before_free_memory;
    }

    leak_checker::
    operator bool() const noexcept 
    {
        if(!_cached)
        {
            // (void)cuda_set_device(_device_id);

            auto result = cuda_mem_get_info();

            auto after_free_memory = result.value().first;
            return after_free_memory == _before_free_memory;
        }

        return _after_free_memory == _before_free_memory;
    }

    size_t
    leak_checker::
    initial_memory() const noexcept
    {
        return _before_free_memory;
    }
}
