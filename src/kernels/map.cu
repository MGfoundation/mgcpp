
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/bits/map.cuh>
#include <cmath>

#define BLK 64

namespace mgcpp
{
#define MGCPP_DEFINE_MAP_FUNCTION(fname, cudaname) \
    __global__ void \
    mgblas_S ## fname ## _impl(float* x, size_t n) \
    { \
        int const id = blockIdx.x * blockDim.x + threadIdx.x; \
        if(id < n) \
            x[id] = cudaname ## f(x[id]); \
    } \
    mgblas_error_t \
    mgblas_S ## fname (float* x, size_t n) \
    { \
        if (n == 0) \
            return invalid_range; \
        int grid_size = \
            static_cast<int>( \
                ceil(static_cast<float>(n)/ BLK)); \
        mgblas_S ## fname ##_impl<<<BLK, grid_size>>>(x, n); \
        return success; \
    } \
    __global__ void \
    mgblas_D ## fname ## _impl(double* x, size_t n) \
    { \
        int const id = blockIdx.x * blockDim.x + threadIdx.x; \
        if(id < n) \
            x[id] = cudaname(x[id]); \
    } \
    mgblas_error_t \
    mgblas_D ## fname (double* x, size_t n) \
    { \
        if (n == 0) \
            return invalid_range; \
        int grid_size = \
            static_cast<int>( \
                ceil(static_cast<float>(n)/ BLK)); \
        mgblas_D ## fname ##_impl<<<BLK, grid_size>>>(x, n); \
        return success; \
    }

    MGCPP_DEFINE_MAP_FUNCTION(vab, fabs)

    // define trig functions
    MGCPP_DEFINE_MAP_FUNCTION(vsin, sin)
    MGCPP_DEFINE_MAP_FUNCTION(vcos, cos)
    MGCPP_DEFINE_MAP_FUNCTION(vtan, tan)
    MGCPP_DEFINE_MAP_FUNCTION(vsinh, sinh)
    MGCPP_DEFINE_MAP_FUNCTION(vcosh, cosh)
    MGCPP_DEFINE_MAP_FUNCTION(vtanh, tanh)

    __device__ float reluf(float f)
    {
        return fmaxf(f, 0.f);
    }

    __device__ double relu(double f)
    {
        return fmax(f, 0.);
    }

    MGCPP_DEFINE_MAP_FUNCTION(vrelu, relu)

#undef MGCPP_DEFINE_MAP_FUNCTION
}
