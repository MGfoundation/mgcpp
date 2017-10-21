
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cublas/blaslike_ext.hpp>
#include <mgcpp/system/exception.hpp>

#include <cublas_v2.h>

namespace mgcpp
{
    template<>
    inline outcome::result<void>
    cublas_geam(cublasHandle_t handle,
                cublasOperation_t transa,
                cublasOperation_t transb,
                size_t m, size_t n,
                float const* alpha,
                float const* A, size_t lda,
                float const* beta,
                float const* B, size_t ldb,
                float *C, size_t ldc) noexcept
    {
        std::error_code err = cublasSgeam(handle,
                                          transa, transb,
                                          m, n,
                                          alpha,
                                          A, lda,
                                          beta,
                                          B, ldb,
                                          C, ldc);

        if(err)
            return err;
        else
            return outcome::success();
    }

    inline outcome::result<void>
    cublas_geam(cublasHandle_t handle,
                cublasOperation_t transa,
                cublasOperation_t transb,
                size_t m, size_t n,
                double const* alpha,
                double const* A, size_t lda,
                double const* beta,
                double const* B, size_t ldb,
                double* C, size_t ldc) noexcept
    {
        std::error_code err = cublasDgeam(handle,
                                          transa, transb,
                                          m, n,
                                          alpha,
                                          A, lda,
                                          beta,
                                          B, ldb,
                                          C, ldc);

        if(err)
            return err;
        else
            return outcome::success();
    }
}
