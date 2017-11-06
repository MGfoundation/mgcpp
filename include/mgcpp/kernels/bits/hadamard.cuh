
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_KERNELS_VECTOR_HADAMARD_HPP_
#define _MGCPP_KERNELS_VECTOR_HADAMARD_HPP_

#include <cuda_fp16.h>

#include <mgcpp/kernels/kernel_status.hpp>

namespace mgcpp
{
    kernel_status_t
    mgblas_Svhp(float const* x, float const* y,
		float* z, size_t size);

    kernel_status_t
    mgblas_Dvhp(double const* x, double const* y,
		double* z, size_t size); 

    kernel_status_t
    mgblas_Hvhp(__half const* x, __half const* y,
		__half* z, size_t size);
}

#endif