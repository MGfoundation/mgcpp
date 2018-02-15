
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_MAT_TRANS_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_MAT_TRANS_EXPR_HPP_

#include <mgcpp/expressions/dmat_expr.hpp>

namespace mgcpp {
template <typename Expr>
struct dmat_trans_expr : dmat_expr<dmat_trans_expr<Expr>> {
  using expr_type = typename std::decay<Expr>::type;
  using result_type = typename expr_type;

  Expr const& _expr;

  inline dmat_trans_expr(Expr const& mat) noexcept;

  inline decltype(auto) eval(bool eval_trans = true);
};

template <typename Expr>
inline decltype(auto) eval(dmat_trans_expr<Expr> trans_expr,
                           bool eval_trans = true);

template <typename Expr>
inline dmat_trans_expr<Expr> trans(
    dmat_expr<Expr> const& dense_mat_expr) noexcept;

template <typename Expr>
inline dmat_trans_expr<Expr> operator!(
    dmat_expr<Expr> const& dense_mat_expr) noexcept;
}  // namespace mgcpp

#include <dmat_trans_expr.tpp>
#endif
