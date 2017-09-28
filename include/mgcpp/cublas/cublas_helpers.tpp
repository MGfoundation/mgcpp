
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <outcome.hpp>

#include <mgcpp/cublas/cublas_helpers.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>

namespace outcome = OUTCOME_V2_NAMESPACE;

namespace mgcpp
{
    template<typename ElemType>
    outcome::result<void>
    cublas_set_matrix(size_t rows, size_t cols,
                      ElemType const* A, size_t leading_dim_A,
                      ElemType* B, size_t leading_dim_B)
    {
        std::error_code status =
            cublasSetMatrix(rows, cols, sizeof(ElemType),
                            A, leading_dim_A, B, leading_dim_B);

        if(status != make_error_condition(status_t::success))
            return status;
        else
            return outcome::success();
    }


    template<typename ElemType>
    std::error_code
    cublas_set_matrix(size_t rows, size_t cols,
                      ElemType const* A, ElemType* B)
    {
        std::error_code status =
            cublasSetMatrix(rows, cols, sizeof(ElemType),
                            A, rows, B, rows);

        if(status != make_error_condition(status_t::success))
            return status;
        else
            return outcome::success();
    }
}
