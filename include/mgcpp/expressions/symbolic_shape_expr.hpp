#ifndef SYMBOLIC_SHAPE_EXPR_HPP
#define SYMBOLIC_SHAPE_EXPR_HPP

#include <mgcpp/expressions/generic_expr.hpp>

namespace mgcpp {

template <typename Expr>
struct shape_expr : expression<Expr> {};

struct symbolic_shape_expr_type {};

template <typename Expr>
using symbolic_shape_expr = generic_expr<symbolic_shape_expr_type,
                                         0,
                                         shape_expr,
                                         typename Expr::result_type::shape_type,
                                         1,
                                         Expr>;


/*
 * Obtain the dynamic shape of this expression
 */
template <typename Expr>
inline symbolic_shape_expr<Expr> sym_shape(Expr const& expr) {
  return symbolic_shape_expr<Expr>(expr);
}

}

#endif // SYMBOLIC_SHAPE_EXPR_HPP
