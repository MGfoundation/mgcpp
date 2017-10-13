
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUBLAS_BLASLIKE_EXT_HPP_
#define _MGCPP_CUBLAS_BLASLIKE_EXT_HPP_

#include <cstdlib>

#include <outcome.hpp>

#include <cublas_v2.h>

namespace outcome = OUTCOME_V2_NAMESPACE;

namespace mgcpp
{
    template<typename T>
    inline outcome::result<void>
    cublas_geam(cublasHandle_t handle,
                cublasOperation_t transa, cublasOperation_t transb,
                int m, int n,
                const float *alpha,
                const float *A, int lda,
                const float *beta,
                const float *B, int ldb,
                float *C, int ldc) noexcept;
}
