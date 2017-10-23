
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/system/cublas_error.hpp>

#include <cuda_runtime.h>

namespace mgcpp
{
    class cublas_error_category_t
        : public std::error_category
    {
    public:
        const char*
        name() const noexcept override;

        std::string
        message(int ev) const override;
    } cublas_error_category;


    const char*
    cublas_error_category_t::
    name() const noexcept
    {
        return "cublas";
    }

    std::string
    cublas_error_category_t::
    message(int ev) const
    {
        switch(static_cast<cublas_error_t>(ev))
        {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS: Operation completed successfully";
            break;

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUABLS_STATUS_NOT_INITIALIZED: The cuBLAS library was not initialized.";
            break;

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED: Resource allocation failed inside the cuBLAS";
            break;

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE: An unsupported value or parameter was passed to";
            break;

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH: The function requires a feature absent from the device architecture";
            break;

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR: An access to GPU memory space failed";
            break;

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED: The GPU Program failed to execute.";
            break;

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED: The operation is not supported by cublas.";
            break;

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "BLAS_STATUS_INTERNAL_ERROR: An internal cuBLAS operation failed";
            break;

#if CUDART_VERSION >= 6500
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR: The cuBlas license is not valid.";
            break;
#endif

        }
        return "";
    }
}

std::error_code
make_error_code(mgcpp::cublas_error_t err) noexcept
{
    return {static_cast<int>(err), mgcpp::cublas_error_category};
}
