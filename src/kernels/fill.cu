
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

     }
}
