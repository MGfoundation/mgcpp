
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cublas/blas_lv1.hpp>
#include <mgcpp/system/error_code.hpp>

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


    inline outcome::result<void>
    cublas_axpy(cublasHandle_t handle, size_t n,
                float const* alpha,
                float const* x, size_t incx,
                float* y, size_t incy) noexcept
    {
        std::error_code status = cublasSaxpy(handle, n,
                                             alpha,
                                             x, incx,
                                             y, incy); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    inline outcome::result<void>
    cublas_axpy(cublasHandle_t handle, size_t n,
                double const* alpha,
                double const* x, size_t incx,
                double* y, size_t incy) noexcept
    {
        
        std::error_code status = cublasDaxpy(handle, n,
                                             alpha,
                                             x, incx,
                                             y, incy); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    inline outcome::result<void>
    cublas_scal(cublasHandle_t handle, size_t n,
                float const* alpha,
                float* vec, size_t incvec) noexcept
    {
        std::error_code status = cublasSscal(handle, n,
                                             alpha,
                                            vec, incvec); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    inline outcome::result<void>
    cublas_scal(cublasHandle_t handle, size_t n,
                double const* alpha,
                double* vec, size_t incvec) noexcept
    {
        std::error_code status = cublasDscal(handle, n,
                                             alpha,
                                            vec, incvec); 

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }
}
