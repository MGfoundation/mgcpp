#ifndef _MGCPP_EXPRESSIONS_DMAT_TRANS_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_TRANS_EXPR_HPP_

#include <mgcpp/matrix/forward.hpp>
#include <mgcpp/expressions/dmat_expr.hpp>

namespace mgcpp {
template <typename Expr>
struct dmat_trans_expr : public dmat_expr<dmat_trans_expr<Expr>> {
  using expr_type = typename std::decay<Expr>::type;

  using result_type = typename expr_type::result_type;

  Expr _mat;

  inline dmat_trans_expr(Expr const& mat) noexcept;
  inline dmat_trans_expr(Expr&& mat) noexcept;

  inline decltype(auto) eval() const;
};

template <typename Expr>
inline decltype(auto) eval(dmat_trans_expr<Expr> const& expr);

template <typename Expr>
inline dmat_trans_expr<Expr> trans(dmat_expr<Expr> const& expr) noexcept;
}

#endif
