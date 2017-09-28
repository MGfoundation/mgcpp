
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda/memory_management.hpp>
#include <mgcpp/system/error_code.hpp>

#include <cuda_runtime.h>

namespace mgcpp
{
    outcome::result<std::pair<free_mem_t, total_mem_t>>
    cuda_mem_get_info() noexcept
    {
        size_t free_memory;
        size_t total_memory;

        std::error_code status =
            cudaMemGetInfo(&free_memory, &total_memory);

        if(status != make_error_condition(status_t::success))
            return status;
        else
            return std::make_pair(free_memory, total_memory);
    }
}
