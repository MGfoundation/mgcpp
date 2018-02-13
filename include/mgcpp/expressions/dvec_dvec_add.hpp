
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DVEC_DVEC_ADD_HPP_
#define _MGCPP_EXPRESSIONS_DVEC_DVEC_ADD_HPP_

#include <mgcpp/expressions/dvec_expr.hpp>
#include <mgcpp/expressions/expr_eval.hpp>
#include <mgcpp/vector/device_vector.hpp>

namespace mgcpp {
template <typename LhsExpr, typename RhsExpr>
struct dvec_dvec_add_expr
    : public dvec_expr<dvec_dvec_add_expr<LhsExpr, RhsExpr>> {
  using lhs_expr_type = typename std::decay<LhsExpr>::type;
  using rhs_expr_type = typename std::decay<RhsExpr>::type;

  using result_type = typename lhs_expr_type::result_type;

  LhsExpr const& _lhs;
  RhsExpr const& _rhs;

  inline dvec_dvec_add_expr(LhsExpr const& lhs, RhsExpr const& rhs) noexcept;

  inline decltype(auto) eval() const;
};

template <typename LhsExpr, typename RhsExpr>
inline decltype(auto) eval(dvec_dvec_add_expr<LhsExpr, RhsExpr> const& expr);

template <typename LhsExpr, typename RhsExpr>
inline dvec_dvec_add_expr<LhsExpr, RhsExpr> operator+(
    dvec_expr<LhsExpr> const& lhs,
    dvec_expr<RhsExpr> const& rhs) noexcept;

template <typename LhsExpr, typename RhsExpr>
inline dvec_dvec_add_expr<LhsExpr, RhsExpr> add(
    dvec_expr<LhsExpr> const& lhs,
    dvec_expr<RhsExpr> const& rhs) noexcept;
}  // namespace mgcpp

#include <mgcpp/expressions/dvec_dvec_add.tpp>
#endif  // _MGCPP_EXPRESSIONS_DVEC_DVEC_ADD_HPP_
