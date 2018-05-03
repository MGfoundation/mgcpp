
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DMAT_DVEC_MULT_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_DVEC_MULT_EXPR_HPP_

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/dvec_expr.hpp>
#include <mgcpp/expressions/generic_op.hpp>

namespace mgcpp {

template <typename MatExpr, typename VecExpr>
using mat_vec_mult_op = binary_op<'m', dvec_expr, typename VecExpr::result_type, MatExpr, VecExpr>;

/** Returns a dense matrix vector product expression.
 * \param lhs the left-hand side dense matrix
 * \param rhs the right-hand side dense vector
 */
template <typename MatExpr, typename VecExpr>
inline mat_vec_mult_op<MatExpr, VecExpr> operator*(
    dmat_expr<MatExpr> const& mat,
    dvec_expr<VecExpr> const& vec) noexcept;

/** Returns a dense matrix add expression.
 * \param lhs the left-hand side dense matrix
 * \param rhs the right-hand side dense vector
 */
template <typename MatExpr, typename VecExpr>
inline mat_vec_mult_op<MatExpr, VecExpr> mult(
    dmat_expr<MatExpr> const& mat,
    dvec_expr<VecExpr> const& vec) noexcept;
}  // namespace mgcpp

#endif
