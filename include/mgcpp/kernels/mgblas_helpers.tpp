
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/bits/fill.cuh>
#include <mgcpp/kernels/bits/convert.cuh>
#include <mgcpp/kernels/mgblas_helpers.hpp>
#include <mgcpp/system/error_code.hpp>

#include <complex>

namespace mgcpp
{
    template<>
    inline outcome::result<void>
    mgblas_fill(float* arr, float value, size_t n)
    {
        std::error_code status = mgblas_Sfill(arr, value, n); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    template<>
    inline outcome::result<void>
    mgblas_fill(double* arr, double value, size_t n)
    {
        std::error_code status = mgblas_Dfill(arr, value, n); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();

    }

    template<>
    inline outcome::result<void>
    mgblas_fill(cuComplex* arr, cuComplex value, size_t n)
    {
        std::error_code status = mgblas_Cfill(arr, value, n);

        if(status != status_t::success)
            return status;
        else
            return outcome::success();

    }

    template<>
    inline outcome::result<void>
    mgblas_fill(cuDoubleComplex* arr, cuDoubleComplex value, size_t n)
    {
        std::error_code status = mgblas_Zfill(arr, value, n);

        if(status != status_t::success)
            return status;
        else
            return outcome::success();

    }

#ifdef USE_HALF
    template<>
    inline outcome::result<void>
    mgblas_fill(__half* arr, __half value, size_t n)
    {
        std::error_code status = mgblas_Hfill(arr, value, n);

        if(status != status_t::success)
            return status;
        else
            return outcome::success();

    }

    template<>
    inline outcome::result<void>
    mgblas_convert_copy(__half const* from, float* to, size_t n)
    {
        std::error_code status = mgblas_HFconvert(from, to, n);

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    template<>
    inline outcome::result<void>
    mgblas_convert_copy(float const* from, __half* to, size_t n)
    {
        std::error_code status = mgblas_FHconvert(from, to, n);

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }
#endif
}
