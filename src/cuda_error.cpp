
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda/internal/cuda_error.hpp>

#include <cuda_runtime.h>

namespace mgcpp
{
    const char*
    internal::
    cuda_error_string(cuda_error_t err)
    {
        return cudaGetErrorString(
            static_cast<cudaError_t>(err));
    }
}
