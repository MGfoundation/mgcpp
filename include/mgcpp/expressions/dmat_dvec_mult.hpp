
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DMAT_DVEC_MULT_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_DVEC_MULT_EXPR_HPP_

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/dvec_expr.hpp>
#include <mgcpp/expressions/generic_expr.hpp>

namespace mgcpp {

template <typename LhsExpr, typename RhsExpr>
struct dmat_dvec_mult_expr : binary_expr<dmat_dvec_mult_expr<LhsExpr, RhsExpr>,
                                         dvec_expr,
                                         typename RhsExpr::result_type,
                                         LhsExpr,
                                         RhsExpr> {
  using binary_expr<dmat_dvec_mult_expr<LhsExpr, RhsExpr>,
                    dvec_expr,
                    typename RhsExpr::result_type,
                    LhsExpr,
                    RhsExpr>::generic_expr;

  template <typename GradsType>
  auto grad(dvec_expr<GradsType> const& grads) const;
};

/** Returns a dense matrix vector product expression.
 * \param lhs the left-hand side dense matrix
 * \param rhs the right-hand side dense vector
 */
template <typename MatExpr, typename VecExpr>
inline dmat_dvec_mult_expr<MatExpr, VecExpr> operator*(
    dmat_expr<MatExpr> const& mat,
    dvec_expr<VecExpr> const& vec) noexcept;

/** Returns a dense matrix add expression.
 * \param lhs the left-hand side dense matrix
 * \param rhs the right-hand side dense vector
 */
template <typename MatExpr, typename VecExpr>
inline dmat_dvec_mult_expr<MatExpr, VecExpr> mult(
    dmat_expr<MatExpr> const& mat,
    dvec_expr<VecExpr> const& vec) noexcept;
}  // namespace mgcpp

#include <mgcpp/expressions/dmat_dvec_mult.tpp>
#endif
