
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUBLAS_CUBLAS_HELPERS_HPP_
#define _MGCPP_CUBLAS_CUBLAS_HELPERS_HPP_

#include <outcome.hpp>

#include <system_error>

namespace outcome = OUTCOME_V2_NAMESPACE;

namespace mgcpp
{
    template<typename ElemType>
    outcome::result<void>
    cublas_set_matrix(size_t rows, size_t cols,
                      ElemType const* A, size_t leading_dim_A,
                      ElemType* B, size_t leading_dim_B);

    template<typename ElemType>
    outcome::result<void>
    cublas_set_matrix(size_t rows, size_t cols,
                      ElemType const* A, ElemType* B);
}

#endif
