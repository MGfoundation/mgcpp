#ifndef _MGCPP_KERNELS_CONVOLUTION_CUH
#define _MGCPP_KERNELS_CONVOLUTION_CUH

#include <cuda_fp16.h>
#include <mgcpp/kernels/kernel_status.hpp>

namespace mgcpp
{
    kernel_status_t
    mgblas_convolution(float* result, float const& f,
                       size_t dim_fx, size_t dim_fy,
                       float const& g);
}

#endif