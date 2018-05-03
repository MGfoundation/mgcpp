
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DMAT_DMAT_MULT_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_DMAT_MULT_EXPR_HPP_

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/generic_op.hpp>

namespace mgcpp {

template <typename LhsExpr, typename RhsExpr>
using mat_mat_mult_op = binary_op<'*', dmat_expr, typename LhsExpr::result_type, LhsExpr, RhsExpr>;

/** Returns a dense matrix product expression.
 * \param lhs the left-hand side dense matrix
 * \param rhs the right-hand side dense matrix
 */
template <typename LhsExpr, typename RhsExpr>
inline mat_mat_mult_op<LhsExpr, RhsExpr> operator*(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept;

/** Returns a dense matrix product expression.
 * \param lhs the left-hand side dense matrix
 * \param rhs the right-hand side dense matrix
 */
template <typename LhsExpr, typename RhsExpr>
inline mat_mat_mult_op<LhsExpr, RhsExpr> mult(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept;
}  // namespace mgcpp

#endif
