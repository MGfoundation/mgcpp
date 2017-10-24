
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUBLAS_BLAS_LV3_HPP_
#define _MGCPP_CUBLAS_BLAS_LV3_HPP_

#include <cstdlib>

#include <outcome.hpp>

namespace outcome = OUTCOME_V2_NAMESPACE;

namespace mgcpp
{
    template<typename T>
    inline outcome::result<void>
    cublas_dot(cublasHandle_t handle, size_t n,
               T const* x, size_t incx,
               T const* y, size_t incy,
               T* result) noexcept;
}

#include <mgcpp/cublas/blas_lv1.tpp>
#endif
