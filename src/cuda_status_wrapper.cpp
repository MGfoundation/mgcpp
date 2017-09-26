
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda/internal/status_wrapper.hpp>

#include <cuda_runtime.h>

namespace mgcpp
{
    internal::cuda_error_t
    internal::
    cuda_mem_get_info(size_t* free, size_t* total)
    {
        return static_cast<cuda_error_t>(cudaMemGetInfo(free, total));
    }
}
