#ifndef MGCPP_EXPRESSIONS_DVEC_DVEC_OUTER_HPP
#define MGCPP_EXPRESSIONS_DVEC_DVEC_OUTER_HPP

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/dvec_expr.hpp>
#include <mgcpp/expressions/generic_expr.hpp>

namespace mgcpp {

struct dvec_dvec_outer_expr_type;

template <typename LhsExpr, typename RhsExpr>
using dvec_dvec_outer_expr =
    binary_expr<dvec_dvec_outer_expr_type,
                dmat_expr,
                device_matrix<typename LhsExpr::result_type::value_type>,
                LhsExpr,
                RhsExpr>;

template <typename LhsExpr, typename RhsExpr>
auto outer(dvec_expr<LhsExpr> const& lhs,
           dvec_expr<RhsExpr> const& rhs) noexcept {
  return dvec_dvec_outer_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
}

}  // namespace mgcpp

#endif
