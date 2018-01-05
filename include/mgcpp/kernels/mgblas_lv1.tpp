
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/bits/hadamard.cuh>
#include <mgcpp/kernels/bits/absolute.cuh>
#include <mgcpp/kernels/bits/reduce.cuh>
#include <mgcpp/kernels/mgblas_lv1.hpp>
#include <mgcpp/system/mgblas_error.hpp>

namespace mgcpp
{
    inline outcome::result<void>
    mgblas_vhp(float const* x, float const* y,
               float* z, size_t n)
    {
        std::error_code status = mgblas_Svhp(x, y, z, n); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    inline outcome::result<void>
    mgblas_vhp(double const* x, double const* y,
               double* z, size_t n)
    {
        std::error_code status = mgblas_Dvhp(x, y, z, n); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    inline outcome::result<void>
    mgblas_vab(float* x, size_t n)
    {
        std::error_code status = mgblas_Svab(x, n); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    inline outcome::result<void>
    mgblas_vab(double* x, size_t n)
    {
        std::error_code status = mgblas_Dvab(x, n); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    inline outcome::result<void>
    mgblas_vpr(float const* x, float* y, size_t n)
    {
        std::error_code status = mgblas_Svpr(x, y, n); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    inline outcome::result<void>
    mgblas_vpr(double const* x, double* y, size_t n)
    {
        std::error_code status = mgblas_Dvpr(x, y, n); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }
}
