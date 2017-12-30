
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/bits/fill.cuh>
#include <cmath>

#define BLK 64

namespace mgcpp
{
    __global__  void
    mgblas_Sfill_impl(float* arr, float value, size_t n)
    {
	int const id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float shared[64];

	if(id >= n)
	    return;

	shared[threadIdx.x] = value;
	__syncthreads();

	arr[id] = shared[threadIdx.x];
    }

    __global__  void
    mgblas_Dfill_impl(double* arr, double value, size_t n)
    {
	int const id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ double shared[64];

	if(id >= n)
	    return;

	shared[threadIdx.x] = value;
	__syncthreads();

	arr[id] = shared[threadIdx.x];
    }

    __global__  void
    mgblas_Cfill_impl(cuComplex* arr, cuComplex value, size_t n)
    {
        int const id = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ cuComplex shared[64];

        if(id >= n)
        return;

        shared[threadIdx.x] = value;
        __syncthreads();

        arr[id] = shared[threadIdx.x];
    }

    __global__  void
    mgblas_Zfill_impl(cuDoubleComplex* arr, cuDoubleComplex value, size_t n)
    {
        int const id = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ cuDoubleComplex shared[64];

        if(id >= n)
        return;

        shared[threadIdx.x] = value;
        __syncthreads();

        arr[id] = shared[threadIdx.x];
    }

    // __global__  void
    // mgblas_Hfill_impl(__half* arr, __half value, size_t n)
    // {
    // 	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 	for (size_t i = idx;
    // 	     i < n;
    // 	     i += gridDim.x * blockDim.x)
    // 	{
    // 	    arr[i] = value;
    // 	}
    // }

    kernel_status_t
    mgblas_Sfill(float* arr, float value, size_t n)
    {
	int grid_size = static_cast<int>(
	    ceil(static_cast<float>(n)/ BLK ));
	mgblas_Sfill_impl<<<BLK, grid_size>>>(arr, value, n);

	return success;
    }

    kernel_status_t
    mgblas_Dfill(double* arr, double value, size_t n)
    {
	int grid_size = static_cast<int>(
	    ceil(static_cast<float>(n)/ BLK ));
	mgblas_Dfill_impl<<<BLK, grid_size>>>(arr, value, n);

	return success;
    }

    kernel_status_t
    mgblas_Cfill(cuComplex* arr, cuComplex value, size_t n)
    {
        int grid_size = static_cast<int>(
            ceil(static_cast<float>(n)/ BLK ));
        mgblas_Cfill_impl<<<BLK, grid_size>>>(arr, value, n);

        return success;
    }

    kernel_status_t
    mgblas_Zfill(cuDoubleComplex* arr, cuDoubleComplex value, size_t n)
    {
        int grid_size = static_cast<int>(
            ceil(static_cast<float>(n)/ BLK ));
        mgblas_Zfill_impl<<<BLK, grid_size>>>(arr, value, n);

        return success;
    }

    // kernel_status_t
    // mgblas_Hfill(__half* arr, __half value, size_t n)
    // {
    // 	int grid_size = static_cast<int>(
    // 	    ceil(static_cast<float>(n)/ BLK ));
    // 	mgblas_Hfill_impl<<<BLK, grid_size>>>(arr, value, n);

    // 	return success;
    // }
}
