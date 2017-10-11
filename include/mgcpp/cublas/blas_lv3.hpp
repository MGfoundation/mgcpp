
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUBLAS_BLAS_LV3_HPP_
#define _MGCPP_CUBLAS_BLAS_LV3_HPP_

#include <cstdlib>

#include <outcome.hpp>

#include <cublas_v2.h>

namespace outcome = OUTCOME_V2_NAMESPACE;

namespace mgcpp
{
    template<typename T>
    inline outcome::result<void>
    cublas_gemm(cublasHandle_t handle,
                cublasOperation_t transa, cublasOperation_t transb,
                size_t m, size_t n, size_t k,
                T const* alpha,
                T const* A, size_t lda,
                T const* B, size_t ldb,
                T const* beta,
                T* C, size_t ldc) noexcept;
}

#endif
