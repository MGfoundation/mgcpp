#ifndef _MGCPP_KERNELS_BITS_CONVERT_CUH_
#define _MGCPP_KERNELS_BITS_CONVERT_CUH_

#include <mgcpp/kernels/mgblas_error_code.hpp>

#include <cuComplex.h>
#include <cuda_fp16.h>

namespace mgcpp
{
    mgblas_error_t
    mgblas_HFconvert(__half const* from, float* to, size_t n);

    mgblas_error_t
    mgblas_FHconvert(float const* from, __half* to, size_t n);
}

#endif
