
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUBLAS_BLAS_LV1_HPP_
#define _MGCPP_CUBLAS_BLAS_LV1_HPP_

#include <cstdlib>

#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

#include <cublas_v2.h>

    namespace mgcpp
{
    template<typename T>
    inline outcome::result<void>
    cublas_dot(cublasHandle_t handle, size_t n,
               T const* x, size_t incx,
               T const* y, size_t incy,
               T* result) noexcept;

    template<typename T>
    inline outcome::result<void>
    cublas_axpy(cublasHandle_t handle, size_t n,
                T const* alpha,
                T const* x, size_t incx,
                T* y, size_t incy) noexcept;

    template<typename T>
    inline outcome::result<void>
    cublas_scal(cublasHandle_t handle, size_t n,
                T const* alpha,
                T* vec, size_t incvec) noexcept;
}

#include <mgcpp/cublas/blas_lv1.tpp>
#endif
