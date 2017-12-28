
#ifndef _MGCPP_KERNELS_BITS_FFT_CUH_
#define _MGCPP_KERNELS_BITS_FFT_CUH_

#include <mgcpp/kernels/kernel_status.hpp>
#include <cuComplex.h>

namespace mgcpp
{
    kernel_status_t mgblas_Cfft(cuComplex const *x, cuComplex *y, size_t n, bool is_inv);
}

#endif
