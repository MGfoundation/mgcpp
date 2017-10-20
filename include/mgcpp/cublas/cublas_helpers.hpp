
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
    outcome::result<ElemType*>
    cublas_set_matrix(size_t rows, size_t cols,
                      ElemType const* from_host, size_t ld_from,
                      ElemType* to_gpu, size_t ld_to);

    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_set_matrix(size_t rows, size_t cols,
                      ElemType const* from_host, ElemType* to_gpu);

    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_get_matrix(size_t rows, size_t cols,
                      ElemType const* from_gpu, size_t ld_from,
                      ElemType* to_host, size_t ld_to);

    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_get_matrix(size_t rows, size_t cols,
                      ElemType const* from_gpu, ElemType* to_host);

    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_set_vector(size_t size,
                      ElemType const* from_host, size_t spacing_host,
                      ElemType* to_gpu, size_t spacing_gpu);

    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_set_vector(size_t size,
                      ElemType const* from_host, ElemType* to_gpu );
}

#include <mgcpp/cublas/cublas_helpers.tpp>
#endif
