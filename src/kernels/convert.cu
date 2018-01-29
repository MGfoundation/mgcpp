
#include <mgcpp/kernels/bits/convert.cuh>

#define BLK 64

namespace mgcpp
{
    __global__ void
    mgblas_HFconvert_impl(__half const* from, float* to, size_t n)
    {
        int const id = blockIdx.x * blockDim.x + threadIdx.x;

        if(id >= n)
        return;

        to[id] = __half2float(from[id]);
    }

    __global__ void
    mgblas_FHconvert_impl(float const* from, __half* to, size_t n)
    {
        int const id = blockIdx.x * blockDim.x + threadIdx.x;

        if(id >= n)
        return;

        to[id] = __float2half(from[id]);
    }

    mgblas_error_t
    mgblas_HFconvert(__half const* from, float* to, size_t n)
    {
        int grid_size = static_cast<int>(
            ceil(static_cast<float>(n)/ BLK ));
        mgblas_HFconvert_impl<<<BLK, grid_size>>>(from, to, n);

        return success;
    }

    mgblas_error_t
    mgblas_FHconvert(float const* from, __half* to, size_t n)
    {
        int grid_size = static_cast<int>(
            ceil(static_cast<float>(n)/ BLK ));
        mgblas_FHconvert_impl<<<BLK, grid_size>>>(from, to, n);

        return success;
    }
}
