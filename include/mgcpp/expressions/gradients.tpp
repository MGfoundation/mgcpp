#include <mgcpp/expressions/gradients.hpp>

namespace mgcpp {

namespace internal {

/**
 * Build a symbolic graph of the gradient of op wrt ph.
 * returns dy/di = dy/d(op) * d(op)/di, given grads = dy/d(op), where y is a
 * scalar function of i, and i is the input(s) of op.
 */
template <typename Expr, typename GradsType>
inline auto grad(dmat_reduce_sum_expr<Expr> const& expr,
                 scalar_expr<GradsType> const& grads) {
  // returns (dmat)
  return std::make_tuple((~grads) * mgcpp::make_ones_like((~expr).first()));
}

template <typename Expr, typename GradsType>
inline auto grad(dvec_reduce_sum_expr<Expr> const& expr,
                 scalar_expr<GradsType> const& grads) {
  // returns (dvec)
  return std::make_tuple((~grads) * mgcpp::make_ones_like((~expr).first()));
}

template <typename LhsExpr, typename RhsExpr, typename GradsType>
inline auto grad(dmat_dmat_add_expr<LhsExpr, RhsExpr> const&,
                 dmat_expr<GradsType> const& grads) {
  // returns (dmat, dmat)
  return std::make_tuple(~grads, ~grads);
}

template <typename LhsExpr, typename RhsExpr, typename GradsType>
inline auto grad(dvec_dvec_add_expr<LhsExpr, RhsExpr> const&,
                 dvec_expr<GradsType> const& grads) {
  // returns (dvec, dvec)
  return std::make_tuple(~grads, ~grads);
}

template <typename LhsExpr, typename RhsExpr, typename GradsType>
inline auto grad(dmat_dmat_mult_expr<LhsExpr, RhsExpr> const& expr,
                 dmat_expr<GradsType> const& grads) {
  // returns (dmat, dmat)
  return std::make_tuple((~grads) * mgcpp::trans((~expr).second()),
                         mgcpp::trans((~expr).first()) * (~grads));
}
/*
template <typename LhsExpr, typename RhsExpr, typename GradsType>
inline auto grad(dmat_dvec_mult_expr<LhsExpr, RhsExpr> const& expr,
                 dvec_expr<GradsType> const& grads) {
  // returns (dmat, dvec)
  return std::make_tuple(mgcpp::outer(~grads, (~expr).second()),
                         mgcpp::trans((~expr).first()) * (~grads));
}
*/
}  // namespace internal

template <typename GradsExpr, size_t PlaceholderID, typename PhResultType>
inline auto grad_impl(placeholder_node<PlaceholderID, PhResultType>,
                      expression<GradsExpr> const& grads,
                      placeholder_node<PlaceholderID, PhResultType>) {
  return ~grads;
}

template <size_t PlaceholderID1,
          typename PhResultType1,
          typename GradsExpr,
          size_t PlaceholderID2,
          typename PhResultType2>
inline auto grad_impl(placeholder_node<PlaceholderID1, PhResultType1>,
                      expression<GradsExpr> const& grads,
                      placeholder_node<PlaceholderID2, PhResultType2>) {
  return make_zeros_like(~grads);
}

/**
 * Evaluates dy/d(wrt) of subgraph rooted at expr.
 */
template <typename Expr,
          typename GradsExpr,
          size_t PlaceholderID,
          typename PhResultType>
inline auto grad_impl(expression<Expr> const& expr,        // w_i
                      expression<GradsExpr> const& grads,  // w_i bar
                      placeholder_node<PlaceholderID, PhResultType> wrt) {
  auto w_bar = internal::grad(~expr, ~grads);
  return sum_tuple(apply(zip((~expr).operands(), w_bar), [&](auto const& p) {
    return grad_impl(p.first, p.second, wrt);
  }));
}

template <typename Expr, size_t PlaceholderID, typename PhResultType>
inline auto grad(scalar_expr<Expr> const& expr,
                 placeholder_node<PlaceholderID, PhResultType> wrt) {
  return grad_impl(~expr,
                   scalar_one_constant_expr<typename Expr::result_type>(), wrt);
}

}  // namespace mgcpp
