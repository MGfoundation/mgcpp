
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUDA_LIBS_CUBLAS_HPP_
#define _MGCPP_CUDA_LIBS_CUBLAS_HPP_

#include <cstdlib>

#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

#include <cublas_v2.h>

namespace mgcpp {
namespace cublas {
// lv1
template <typename T>
inline outcome::result<void> dot(cublasHandle_t handle,
                                 size_t n,
                                 T const* x,
                                 size_t incx,
                                 T const* y,
                                 size_t incy,
                                 T* result) noexcept;

template <typename T>
inline outcome::result<void> axpy(cublasHandle_t handle,
                                  size_t n,
                                  T const* alpha,
                                  T const* x,
                                  size_t incx,
                                  T* y,
                                  size_t incy) noexcept;

template <typename VectorType, typename ScalarType>
inline outcome::result<void> scal(cublasHandle_t handle,
                                  size_t n,
                                  ScalarType const* alpha,
                                  VectorType* vec,
                                  size_t incvec) noexcept;

// lv2

/// matrix-vector multiplication
/// y = alpha * op(A) * x + beta * y
/// @param handle handle to the cuBLAS library context
/// @param trans operation op(A) that is non- or (conj.) transpose.
/// @param m number of rows of matrix A.
/// @param n number of columns of matrix A.
/// @param alpha scalar used for multiplication.
/// @param A array of dimension (lda, n)
/// @param lda leading dimension of two-dimensional array used to store
/// matrix A. lda must be at least max{1, m}
/// @param x vector at least (1+(n-1)*abs(incx)) elements if
/// transa==CUBLAS_OP_N and at least (1+(m-1)*abs(incx)) elements otherwise.
/// @param incx stride between consecutive elements of x.
/// @param beta scalar used for multiplication, if beta==0 then y does not
/// have to be a valid input.
/// @param y vector at least (1+(m-1)*abs(incy)) elements if
/// transa==CUBLAS_OP_N and at least (1+(n-1)*abs(incy)) elements otherwise.
/// @param incy stride between consecutive elements of y.
template <typename T>
inline outcome::result<void> gemv(cublasHandle_t handle,
                                  cublasOperation_t trans,
                                  int m,
                                  int n,
                                  T const* alpha,
                                  T const* A,
                                  int lda,
                                  T const* x,
                                  int incx,
                                  T const* beta,
                                  T* y,
                                  int incy);

// lv3

template <typename T>
inline outcome::result<void> gemm(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  size_t m,
                                  size_t n,
                                  size_t k,
                                  T const* alpha,
                                  T const* A,
                                  size_t lda,
                                  T const* B,
                                  size_t ldb,
                                  T const* beta,
                                  T* C,
                                  size_t ldc) noexcept;

template <typename T>
inline outcome::result<void> gemm_batched(cublasHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          T const* alpha,
                                          T const* A,
                                          size_t lda,
                                          T const* B,
                                          size_t ldb,
                                          T const* beta,
                                          T* C,
                                          size_t ldc) noexcept;

// blas-like cublas extensions

template <typename T>
inline outcome::result<void> geam(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  size_t m,
                                  size_t n,
                                  const T* alpha,
                                  const T* A,
                                  size_t lda,
                                  const T* beta,
                                  const T* B,
                                  size_t ldb,
                                  T* C,
                                  size_t ldc) noexcept;
}  // namespace cublas
}  // namespace mgcpp

#include <mgcpp/cuda_libs/cublas.tpp>
#endif
