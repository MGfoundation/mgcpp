#include <cuda_runtime.h>

#include <mgcpp/internal_cuda/cuda_stdlib.hpp>

namespace mgcpp
{
    namespace internal
    {

        cuda_error_t internal_cuda_malloc(void* ptr, size_t size)
        {
            return static_cast<cuda_error_t>(cudaMalloc(&ptr, size));
        }

        cuda_error_t internal_cuda_free(void* ptr)
        {
            return static_cast<cuda_error_t>(cudaFree(ptr));
        }
    }
}
