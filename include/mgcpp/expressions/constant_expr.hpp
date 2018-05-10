#ifndef CONSTANT_EXPR_HPP
#define CONSTANT_EXPR_HPP

#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/expressions/dmat_expr.hpp>

namespace mgcpp {

template <typename Expr>
using zeros_like = generic_expr<expression_type,
                                expression_type::ALL_ZEROS,
                                dmat_expr,
                                typename Expr::result_type,
                                0,
                                symbolic_shape_expr<Expr>>;

template <typename Expr>
using ones_like = generic_expr<expression_type,
                               expression_type::ALL_ONES,
                               dmat_expr,
                               typename Expr::result_type,
                               0,
                               symbolic_shape_expr<Expr>>;

}  // namespace mgcpp

#endif  // CONSTANT_EXPR_HPP
