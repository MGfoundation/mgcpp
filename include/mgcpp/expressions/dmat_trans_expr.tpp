
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dmat_trans_expr.hpp>
#include <mgcpp/operations/trans.hpp>

namespace mgcpp {
template <typename Expr>
decltype(auto) dmat_trans_expr<Expr>::eval(bool eval_trans) {
  if (eval_trans) {
    auto pre_trans = eval(_expr);
    auto post_trans = mgcpp::strict::trans(result);
    return post_trans;
  } else {
    auto pre_trans = eval(_expr);
    return dmat_trans_expr<Expr>(pre_trans);
  }
}

template <typename Expr>
dmat_trans_expr(Expr const& mat) noexcept : _expr(mat) {}

template <typename Expr>
decltype(auto) eval(dmat_trans_expr<Expr> trans_expr, bool eval_trans);
{ return tans_expr.eval(eval_trans); }

template <typename Expr>
dmat_trans_expr<Expr> trans(dmat_expr<Expr> const& dense_mat_expr) noexcept {
  return dmat_trans_expr<Expr>(~dense_mat_expr);
}

template <typename Expr>
dmat_trans_expr<Expr> operator!(
    dmat_expr<Expr> const& dense_mat_expr) noexcept {
  return dmat_trans_expr<Expr>(~dense_mat_expr);
}

}  // namespace mgcpp
