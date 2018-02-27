
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_KERNELS_ABSOLUTE_HPP_
#define _MGCPP_KERNELS_ABSOLUTE_HPP_

#include <cstddef>
#include <cuda_fp16.h>

#include <mgcpp/kernels/mgblas_error_code.hpp>

namespace mgcpp
{
    mgblas_error_t
    mgblas_Svab(float* x, size_t n);

    mgblas_error_t
    mgblas_Dvab(double* x, size_t n); 

    // kernel_status_t
    // mgblas_Hvab(__half const* x, __half const* y,
    // 		__half* z, size_t size);

    mgblas_error_t
    mgblas_Svsin(float* x, size_t n);

    mgblas_error_t
    mgblas_Dvsin(double* x, size_t n);

    mgblas_error_t
    mgblas_Svcos(float* x, size_t n);

    mgblas_error_t
    mgblas_Dvcos(double* x, size_t n);

    mgblas_error_t
    mgblas_Svtan(float* x, size_t n);

    mgblas_error_t
    mgblas_Dvtan(double* x, size_t n);

    mgblas_error_t
    mgblas_Svsinh(float* x, size_t n);

    mgblas_error_t
    mgblas_Dvsinh(double* x, size_t n);

    mgblas_error_t
    mgblas_Svcosh(float* x, size_t n);

    mgblas_error_t
    mgblas_Dvcosh(double* x, size_t n);

    mgblas_error_t
    mgblas_Svtanh(float* x, size_t n);

    mgblas_error_t
    mgblas_Dvtanh(double* x, size_t n);

    mgblas_error_t
    mgblas_Svrelu(float* x, size_t n);

    mgblas_error_t
    mgblas_Dvrelu(double* x, size_t n);
}

#endif
