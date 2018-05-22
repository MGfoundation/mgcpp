
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DMAT_DMAT_ADD_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_DMAT_ADD_EXPR_HPP_

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/forward.hpp>
#include <mgcpp/expressions/generic_expr.hpp>

namespace mgcpp {

struct dmat_dmat_add_expr_type;

template <typename LhsExpr, typename RhsExpr>
using dmat_dmat_add_expr = binary_expr<dmat_dmat_add_expr_type,
                                       dmat_expr,
                                       typename LhsExpr::result_type,
                                       LhsExpr,
                                       RhsExpr>;

template <typename LhsExpr, typename RhsExpr>
inline auto operator+(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept;

template <typename LhsExpr, typename RhsExpr>
inline auto add(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept;

}  // namespace mgcpp

#include <mgcpp/expressions/dmat_dmat_add.tpp>
#endif
