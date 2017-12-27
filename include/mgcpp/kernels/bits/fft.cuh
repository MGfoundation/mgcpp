
#ifndef _MGCPP_KERNELS_BITS_FFT_CUH_
#define _MGCPP_KERNELS_BITS_FFT_CUH_

#include <mgcpp/kernels/kernel_status.hpp>
#include <mgcpp/global/complex.hpp>

namespace mgcpp
{
    kernel_status_t mgblas_Cfft(complex<float> const *x, complex<float> *y, size_t n);
}

#endif
