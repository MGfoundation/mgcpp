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
