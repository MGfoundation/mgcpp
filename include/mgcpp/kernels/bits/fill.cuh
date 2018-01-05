
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_KERNELS_BITS_FILL_CUH_
#define _MGCPP_KERNELS_BITS_FILL_CUH_

#include <mgcpp/kernels/mgblas_error_code.hpp>

#include <cuComplex.h>
//#include <cuda_fp16.h>

namespace mgcpp
{
    mgblas_error_t
    mgblas_Sfill(float* arr, float value, size_t n);

    mgblas_error_t
    mgblas_Dfill(double* arr, double value, size_t n);

    mgblas_error_t
    mgblas_Cfill(cuComplex* arr, cuComplex value, size_t n);

    mgblas_error_t
    mgblas_Zfill(cuDoubleComplex* arr, cuDoubleComplex value, size_t n);

    // kernel_status_t
    // mgblas_Hfill(__half* arr, __half value, size_t n);
}

#endif
