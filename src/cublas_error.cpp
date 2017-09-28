
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/system/cublas_error.hpp>

namespace mgcpp
{
    const char*
    internal::cublas_error_category_t::
    name() const noexcept
    {
        return "cublas";
    }

    std::string
    internal::cublas_error_category_t::
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

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "BLAS_STATUS_INTERNAL_ERROR: An internal cuBLAS operation failed";
            break;

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED: The functionnality requested is not supported";
            break;

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR: The functionnality requested requires some license and an error was detected when trying to check the current licensing.";
            break;
        }
        return "";
    }

    internal::cublas_error_category_t internal::cublas_error_category;
}

std::error_code
make_error_code(mgcpp::cublas_error_t err) noexcept
{
    return {static_cast<int>(err),
            mgcpp::internal::cublas_error_category};
}
