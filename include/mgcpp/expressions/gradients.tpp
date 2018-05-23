#include <mgcpp/expressions/gradients.hpp>

namespace mgcpp {
namespace internal {
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
                      expression<GradsExpr> const&,
                      placeholder_node<PlaceholderID2, PhResultType2> wrt) {
  return make_zeros_like(wrt);
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
  /**
   * Build a symbolic graph of the gradient of op wrt ph.
   * returns dy/di = dy/d(op) * d(op)/di, given grads = dy/d(op), where y is a
   * scalar function of i, and i is the input(s) of op.
   */
  auto w_bar = (~expr).grad(~grads);
  return sum_tuple(apply(zip((~expr).operands(), w_bar), [&](auto const& p) {
    return grad_impl(p.first, p.second, wrt);
  }));
}
}  // namespace internal

template <typename Expr, size_t PlaceholderID, typename PhResultType>
inline auto grad(scalar_expr<Expr> const& expr,
                 placeholder_node<PlaceholderID, PhResultType> wrt) {
  return internal::grad_impl(
      ~expr, scalar_one_constant_expr<typename Expr::result_type>(), wrt);
}

}  // namespace mgcpp
