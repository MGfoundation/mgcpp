
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cuda_runtime.h>

#include <mgcpp/cuda/internal/stdlib_wrapper.hpp>

namespace mgcpp
{
    internal::cuda_error_t
    internal::
    cuda_malloc(void** ptr, size_t size)
    {
        return static_cast<cuda_error_t>(cudaMalloc(ptr, size));
    }

    internal::cuda_error_t
    internal::
    cuda_free(void* ptr)
    {
        return static_cast<cuda_error_t>(cudaFree(ptr));
    }
}
