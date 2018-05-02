
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DMAT_DMAT_ADD_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_DMAT_ADD_EXPR_HPP_

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/binary_op.hpp>

namespace mgcpp {

template <typename LhsExpr, typename RhsExpr>
using mat_mat_add_op = binary_op<'+', LhsExpr, RhsExpr, dmat_expr, typename LhsExpr::result_type>;

template <typename LhsExpr, typename RhsExpr>
inline mat_mat_add_op<LhsExpr, RhsExpr> operator+(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept;

template <typename LhsExpr, typename RhsExpr>
inline mat_mat_add_op<LhsExpr, RhsExpr> add(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept;

}  // namespace mgcpp

#endif
