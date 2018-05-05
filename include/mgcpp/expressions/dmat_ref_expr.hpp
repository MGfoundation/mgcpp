
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DMAT_REF_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_REF_EXPR_HPP_

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/eval_context.hpp>
#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/matrix/forward.hpp>

namespace mgcpp {

template <typename Matrix>
using dmat_ref_expr = generic_expr<expression_type,
                                 expression_type::DMAT_REF,
                                 dmat_expr,
                                 Matrix,
                                 1,
                                 Matrix const&>;

template <typename DenseMatrix, typename Type, size_t DeviceId>
inline dmat_ref_expr<DenseMatrix> ref(
    dense_matrix<DenseMatrix, Type, DeviceId> const& mat);
}  // namespace mgcpp

#include <mgcpp/expressions/dmat_ref_expr.tpp>
#endif
