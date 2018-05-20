#include <mgcpp/expressions/dmat_reduce_expr.hpp>

namespace mgcpp {

template <typename Expr>
decltype(auto) reduce_sum(const dmat_expr<Expr>& expr) noexcept {
  return dmat_reduce_sum_expr<Expr>(~expr);
}

template <typename Expr>
decltype(auto) reduce_mean(const dmat_expr<Expr>& expr) noexcept {
  return dmat_reduce_mean_expr<Expr>(~expr);
}

}
