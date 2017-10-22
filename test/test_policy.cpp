
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cuda_runtime.h>

#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>
#include "test_policy.hpp"

namespace mgcpp
{
    test_policy&
    test_policy::
    get_policy()
    {
        static test_policy _policy;
        return _policy;
    }

    test_policy::
    test_policy()
    {
        int device_number = 0;
        std::error_code status = cudaGetDeviceCount(&device_number);
        if(status != status_t::success)
            MGCPP_THROW_SYSTEM_ERROR(status);

        _device_num = static_cast<size_t>(device_number);
    }

    size_t
    test_policy::
    device_num() const noexcept
    {
        return _device_num;
    }
}

