
#ifndef _MGCPP_KERNELS_BITS_FFT_CUH_
#define _MGCPP_KERNELS_BITS_FFT_CUH_

#include <mgcpp/kernels/mgblas_error_code.hpp>
#include <cuComplex.h>

namespace mgcpp
{
    mgblas_error_t
    mgblas_Cfft(cuComplex const *x, cuComplex *y, size_t n, bool is_inv);
}

#endif
