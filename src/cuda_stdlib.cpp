#include <cuda_runtime.h>

#include <mgcpp/cuda/internal/cuda_stdlib.hpp>

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
