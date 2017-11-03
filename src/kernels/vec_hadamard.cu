
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/bits/vec_hadamard.cuh>

namespace mgcpp
{
    __global__ void
    mgcppSvhad_impl(float* x, float* y, float* z, size_t size)
    {
	int const id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < size)
	    z[id] = __fmul_rn(x[id], y[id]);
    }

    __global__ void
    mgcppDvhad_impl(double* x, double* y, double* z, size_t size)
    {
	int const id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < size)
	    z[id] = __dmul_rn(x[id], y[id]);
    }

    __global__ void
    mgcppHvhad_impl(__half* x, __half* y, __half* z, size_t size)
    {
	int const id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < size)
	    z[id] = __hmul(x[id], y[id]);
    }

    kernel_status_t
    mgcppSvhad(float* x, float* y, float* z, size_t size)
    {
	if(size == 0)
	    return invalid_range;

	mgcppSvhad_impl<<<1, 1>>>(x, y, z, size);

	return success;
    }

    kernel_status_t
    mgcppDvhad(double* x, double* y, double* z, size_t size)
    {
	if(size == 0)
	    return invalid_range;
	
	mgcppDvhad_impl<<<1, 1>>>(x, y, z, size);

	return success;
    }

    kernel_status_t
    mgcppHvhad(__half* x, __half* y, __half* z, size_t size)
    {
	if(size == 0)
	    return invalid_range;
	
	mgcppHvhad_impl<<<1, 1>>>(x, y, z, size);

	return success;
    }

}