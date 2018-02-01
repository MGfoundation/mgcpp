
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/bits/absolute.cuh>
#include <cmath>

#define BLK 64

namespace mgcpp
{
    __global__ void
    mgblas_Svab_impl(float* x, size_t n)
    {
        int const id = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ float shared[BLK];

        shared[threadIdx.x] = x[id];
        __syncthreads();

        if(id < n)
            shared[threadIdx.x] = fabsf(shared[threadIdx.x]);
        __syncthreads();

        x[id] = shared[threadIdx.x];
    }

    __global__ void
    mgblas_Dvab_impl(double* x, size_t n)
    {
        int const id = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ double shared[BLK];

        shared[threadIdx.x] = x[id];
        __syncthreads();

        if(id < n)
            shared[threadIdx.x] = fabs(shared[threadIdx.x]);
        __syncthreads();

        x[id] = shared[threadIdx.x];
    }

    // __global__ void
    // mgblas_Hvab_impl(__half* x, size_t n)
    // {
    // 	int const id = blockIdx.x * blockDim.x + threadIdx.x;

    // 	__shared__ __half shared[BLK];

    // 	shared[threadIdx.x] = x[id];
    // 	__syncthreads();

    // 	if(id < n)
    // 	    shared_z[threadIdx.x] = __hmul(shared_x[threadIdx.x], shared_y[threadIdx.x]);
    // 	__syncthreads();

    // 	z[id] = shared_z[threadIdx.x];
    // }

    mgblas_error_t
    mgblas_Svab(float* x, size_t n)
    {
        if(n== 0)
            return invalid_range;

        int grid_size =
            static_cast<int>(
                ceil(static_cast<float>(n)/ BLK ));
        mgblas_Svab_impl<<<BLK, grid_size>>>(x, n);

        return success;
    }

    mgblas_error_t
    mgblas_Dvab(double* x, size_t n)
    {
        if(n== 0)
            return invalid_range;
    
        int grid_size =
            static_cast<int>(
                ceil(static_cast<float>(n)/ BLK ));
        mgblas_Dvab_impl<<<BLK, grid_size>>>(x, n);

        return success;
    }

    // mgblas_error_t
    // mgblas_Hvab(__half* x, size_t n)
    // {
    // 	if(size == 0)
    // 	    return invalid_range;
	
    // 	int grid_size =
    // 	    static_cast<int>(
    // 		ceil(static_cast<float>(size)/ BLK ));
    // 	mgblas_Hvab_impl<<<BLK, grid_size>>>(x, size);

    // 	return success;
    // }
}