#ifndef MGCPP_EXPRESSIONS_DVEC_DVEC_OUTER_HPP
#define MGCPP_EXPRESSIONS_DVEC_DVEC_OUTER_HPP

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/dvec_expr.hpp>
#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/matrix/forward.hpp>

namespace mgcpp {

template <typename LhsExpr, typename RhsExpr>
struct dvec_dvec_outer_expr
    : binary_expr<dvec_dvec_outer_expr<LhsExpr, RhsExpr>,
                  dmat_expr,
                  device_matrix<typename LhsExpr::result_type::value_type,
                                LhsExpr::result_type::device_id,
                                typename LhsExpr::result_type::allocator_type>,
                  LhsExpr,
                  RhsExpr> {
  using binary_expr<
      dvec_dvec_outer_expr<LhsExpr, RhsExpr>,
      dmat_expr,
      device_matrix<typename LhsExpr::result_type::value_type,
                    LhsExpr::result_type::device_id,
                    typename LhsExpr::result_type::allocator_type>,
      LhsExpr,
      RhsExpr>::generic_expr;
};

template <typename LhsExpr, typename RhsExpr>
auto outer(dvec_expr<LhsExpr> const& lhs,
           dvec_expr<RhsExpr> const& rhs) noexcept {
  return dvec_dvec_outer_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
}

}  // namespace mgcpp

#endif
