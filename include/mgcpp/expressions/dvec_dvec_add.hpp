
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DVEC_DVEC_ADD_HPP_
#define _MGCPP_EXPRESSIONS_DVEC_DVEC_ADD_HPP_

#include <mgcpp/expressions/dvec_expr.hpp>
#include <mgcpp/expressions/generic_op.hpp>

namespace mgcpp {
template <typename LhsExpr, typename RhsExpr>
using vec_vec_add_op = binary_op<expression_type::DVEC_DVEC_ADD,
                                 dvec_expr,
                                 typename LhsExpr::result_type,
                                 LhsExpr,
                                 RhsExpr>;

/** Returns a dense vector addition expression.
 * \param lhs the left-hand side dense vector
 * \param rhs the right-hand side dense vector
 */
template <typename LhsExpr, typename RhsExpr>
inline vec_vec_add_op<LhsExpr, RhsExpr> operator+(
    dvec_expr<LhsExpr> const& lhs,
    dvec_expr<RhsExpr> const& rhs) noexcept;

/** Returns a dense vector addition expression.
 * \param lhs the left-hand side dense vector
 * \param rhs the right-hand side dense vector
 */
template <typename LhsExpr, typename RhsExpr>
inline vec_vec_add_op<LhsExpr, RhsExpr> add(
    dvec_expr<LhsExpr> const& lhs,
    dvec_expr<RhsExpr> const& rhs) noexcept;
}  // namespace mgcpp

#endif  // _MGCPP_EXPRESSIONS_DVEC_DVEC_ADD_HPP_
