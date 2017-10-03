
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/thread_context.hpp>

namespace mgcpp
{
    thread_context::
    thread_context(std::initializer_list<size_t> devices_used)
        : _device_managers()
    {
        _device_managers.reserve(devices_used.size());
        for(auto i : devices_used)
        {
            _device_managers.emplace(i, device_manager{i});
        }
    }
    
    cublasHandle_t
    thread_context::
    get_cublas(size_t device_id) 
    {
        return _device_managers[device_id].get_cublas();
    }
}
