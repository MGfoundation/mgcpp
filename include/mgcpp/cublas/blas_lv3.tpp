
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cublas/blas_lv3.hpp>
#include <mgcpp/system/error_code.hpp>

namespace mgcpp
{
    inline outcome::result<void>
    cublas_gemm(cublasHandle_t handle,
                cublasOperation_t trans, cublasOperation_t transb,
               size_t m, size_t n, size_t k,
               float const* alpha,
               float const* A, size_t lda,
               float const* B, size_t ldb,
               float const* beta,
               float* C, size_t ldc) noexcept
    {
        std::error_code status =
            cublasSgemm(handle, transa, transb,
                        m, n, k, alpha, A, lda,
                        B, ldb, beta, C, ldc);

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }

    inline outcome::result<void>
    cublas_gemm(cublasHandle_t handle,
                cublasOperation_t transa, cublasOperation_t transb,
                size_t m, size_t n, size_t k,
                double const* alpha,
                double const* A, size_t lda,
                double const* B, size_t ldb,
                double const* beta,
                double* C, size_t ldc) noexcept
    {
        std::error_code status =
            cublasSgemm(handle, transa, transb,
                        m, n, k, alpha, A, lda,
                        B, ldb, beta, C, ldc);

        if(status != status_t::success)
            return status;
        else
            return outcome::success();
    }
}
