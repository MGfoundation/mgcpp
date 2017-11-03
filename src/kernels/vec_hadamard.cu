
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cuda_runtime.h>

#include <mgcpp/kernels/bits/vec_hadamard.cuh>
#include <mgcpp/kernels/kernel_status.hpp>

namespace mgcpp
{
    __global__ void
    mgcppSvhad(float* x, float* y, float* z, size_t size)
    {
	
    }

    __global__ void
    mgcppDvhad(double* x, double* y, double* z, size_t size)
    {
	
    }

    kernel_status_t
    mgcppSvhad(float* x, float* y, float* z, size_t size)
    {
	if(size == 0)
	    return invalid_range;

	mgcppSvhad<<<>>>();

	return success;
    }

    kernel_status_t
    mgcppDvhad(double* x, double* y, double* z, size_t size)
    {
	if(size == 0)
	    return invalid_range;
	
	mgcppDvhad<<<>>>();

	return success;
    }
}