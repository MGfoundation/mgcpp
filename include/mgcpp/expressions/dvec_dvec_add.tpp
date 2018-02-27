
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dvec_dvec_add.hpp>
#include <mgcpp/operations/add.hpp>

namespace mgcpp {
template <typename LhsExpr, typename RhsExpr>
dvec_dvec_add_expr<LhsExpr, RhsExpr>::dvec_dvec_add_expr(
    LhsExpr const& lhs,
    RhsExpr const& rhs) noexcept
    : _lhs(lhs), _rhs(rhs) {}

template <typename LhsExpr, typename RhsExpr>
dvec_dvec_add_expr<LhsExpr, RhsExpr>::dvec_dvec_add_expr(LhsExpr&& lhs,
                                                         RhsExpr&& rhs) noexcept
    : _lhs(std::move(lhs)), _rhs(std::move(rhs)) {}

template <typename LhsExpr, typename RhsExpr>
typename dvec_dvec_add_expr<LhsExpr, RhsExpr>::result_type
dvec_dvec_add_expr<LhsExpr, RhsExpr>::eval() const {
  auto const& lhs = mgcpp::eval(_lhs);
  auto const& rhs = mgcpp::eval(_rhs);

  return strict::add(lhs, rhs);
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
