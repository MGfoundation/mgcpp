#ifndef CONSTANT_EXPR_HPP
#define CONSTANT_EXPR_HPP

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/expressions/symbolic_shape_expr.hpp>

namespace mgcpp {

struct zeros_mat_expr_type;

template <typename Expr>
using zeros_mat_expr = generic_expr<zeros_mat_expr_type,
                                    0,
                                    dmat_expr,
                                    typename Expr::result_type,
                                    1,
                                    symbolic_shape_expr<Expr>>;

template <typename Expr>
zeros_mat_expr<Expr> make_zeros_like(dmat_expr<Expr> const& expr) {
  return zeros_mat_expr<Expr>(sym_shape(~expr));
}

struct ones_mat_expr_type;

template <typename Expr>
using ones_mat_expr = generic_expr<ones_mat_expr_type,
                                   0,
                                   dmat_expr,
                                   typename Expr::result_type,
                                   1,
                                   symbolic_shape_expr<Expr>>;

template <typename Expr>
ones_mat_expr<Expr> make_ones_like(dmat_expr<Expr> const& expr) {
  return ones_mat_expr<Expr>(sym_shape(~expr));
}

}  // namespace mgcpp

#endif  // CONSTANT_EXPR_HPP
