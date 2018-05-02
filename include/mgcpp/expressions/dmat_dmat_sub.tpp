
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dmat_dmat_sub.hpp>

namespace mgcpp {
template <typename LhsExpr, typename RhsExpr>
mat_mat_add_op<LhsExpr, RhsExpr> operator-(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept {
  auto const& lhs_orig = ~lhs;
  auto const& rhs_orig = (-1) * (~rhs);
  return mat_mat_add_op<LhsExpr, RhsExpr>(lhs_orig, rhs_orig);
}

template <typename LhsExpr, typename RhsExpr>
mat_mat_add_op<LhsExpr, RhsExpr> sub(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept {
  auto const& lhs_orig = ~lhs;
  auto const& rhs_orig = (-1) * (~rhs);
  return mat_mat_add_op<LhsExpr, RhsExpr>(lhs_orig, rhs_orig);
}
}  // namespace mgcpp
