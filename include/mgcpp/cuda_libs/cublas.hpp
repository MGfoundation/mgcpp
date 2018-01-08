
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUDA_LIBS_CUBLAS_HPP_
#define _MGCPP_CUDA_LIBS_CUBLAS_HPP_

#include <cstdlib>

#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

#include <cublas_v2.h>

namespace mgcpp
{
    // lv1
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

    template<typename VectorType, typename ScalarType>
    inline outcome::result<void>
    cublas_scal(cublasHandle_t handle, size_t n,
                ScalarType const* alpha,
                VectorType* vec, size_t incvec) noexcept;

    // lv3

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

    // blas-like cublas extensions

    template<typename T>
    inline outcome::result<void>
    cublas_geam(cublasHandle_t handle,
                cublasOperation_t transa,
                cublasOperation_t transb,
                size_t m, size_t n,
                const T *alpha,
                const T *A, size_t lda,
                const T *beta,
                const T *B, size_t ldb,
                T *C, size_t ldc) noexcept;
}

#include <mgcpp/cuda_libs/cublas.tpp>
#endif
