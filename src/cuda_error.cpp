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
