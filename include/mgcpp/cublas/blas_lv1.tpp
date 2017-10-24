
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cublas/blas_lv1.hpp>
#include <mgcpp/system/error_code.hpp>

#include <cublas_v2.h>

namespace mgcpp
{
    inline outcome::result<void>
    cublas_dot(cublasHandle_t handle, size_t n,
               float const* x, size_t incx,
               float const* y, size_t incy,
               float* result) noexcept
    {
        std::error_code status =
            cublasSdot(handle, n, x, incx, y, incy, result); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    inline outcome::result<void>
    cublas_dot(cublasHandle_t handle, size_t n,
               double const* x, size_t incx,
               double const* y, size_t incy,
               double* result) noexcept
    {
        
        std::error_code status =
            cublasDdot(handle, n, x, incx, y, incy, result); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }
}
