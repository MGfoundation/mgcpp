
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cublas/blaslike_ext.hpp>

#include <mgcpp/system/exception.hpp>

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
                float *C, int ldc) noexcept
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
            return MGCPP_THROW_SYSTEM_ERROR(err);
        else
            return outcome::success();
    }
}
