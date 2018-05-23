
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dvec_dvec_add.hpp>

namespace mgcpp {

template <typename LhsExpr, typename RhsExpr>
template <typename GradsType>
auto dvec_dvec_add_expr<LhsExpr, RhsExpr>::grad(
    dvec_expr<GradsType> const& grads) const {
  return std::make_tuple(~grads, ~grads);
}

template <typename LhsExpr, typename RhsExpr>
dvec_dvec_add_expr<LhsExpr, RhsExpr> operator+(
    dvec_expr<LhsExpr> const& lhs,
    dvec_expr<RhsExpr> const& rhs) noexcept {
  return dvec_dvec_add_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
}

template <typename LhsExpr, typename RhsExpr>
dvec_dvec_add_expr<LhsExpr, RhsExpr> add(
    dvec_expr<LhsExpr> const& lhs,
    dvec_expr<RhsExpr> const& rhs) noexcept {
  return dvec_dvec_add_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
}
}  // namespace mgcpp
