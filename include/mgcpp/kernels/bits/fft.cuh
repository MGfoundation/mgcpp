
#ifndef _MGCPP_KERNELS_BITS_FFT_CUH_
#define _MGCPP_KERNELS_BITS_FFT_CUH_

#include <mgcpp/kernels/kernel_status.hpp>

namespace mgcpp
{
    kernel_status_t mgblas_Srfft(float const *x, float *y, size_t n);
}

#endif
