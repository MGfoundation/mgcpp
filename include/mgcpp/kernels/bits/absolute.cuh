
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_KERNELS_ABSOLUTE_HPP_
#define _MGCPP_KERNELS_ABSOLUTE_HPP_

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
}

#endif