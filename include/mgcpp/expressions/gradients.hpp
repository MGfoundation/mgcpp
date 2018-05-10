#ifndef GRADIENTS_HPP
#define GRADIENTS_HPP

#include <mgcpp/expressions/constant_expr.hpp>
#include <mgcpp/expressions/dmat_dmat_add.hpp>
#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/expressions/placeholder.hpp>

namespace mgcpp {

template <typename Op,
          int PlaceholderID,
          template <typename> class ResultExprType,
          typename ResultType>
inline static auto grad(
    Op const& op,
    placeholder_node<PlaceholderID, ResultExprType, ResultType> ph);

namespace internal {
template <typename LhsExpr,
          typename RhsExpr,
          int PlaceholderID,
          template <typename> class ResultExprType,
          typename ResultType>
inline auto grad(
    dmat_dmat_add_expr<LhsExpr, RhsExpr> const& expr,
    placeholder_node<PlaceholderID, ResultExprType, ResultType> ph) {
  return mgcpp::grad(expr.first(), ph) + mgcpp::grad(expr.second(), ph);
}

template <int PlaceholderID,
          template <typename> class ResultExprType,
          typename ResultType>
inline auto grad(
    placeholder_node<PlaceholderID, ResultExprType, ResultType>,
    placeholder_node<PlaceholderID, ResultExprType, ResultType> ph) {
  return make_ones_like(ph);
}

}  // namespace internal

template <typename Op,
          int PlaceholderID,
          template <typename> class ResultExprType,
          typename ResultType>
inline static auto grad(
    Op const& op,
    placeholder_node<PlaceholderID, ResultExprType, ResultType> ph) {
  return internal::grad(op, ph);
}

}  // namespace mgcpp

#endif  // GRADIENTS_HPP
