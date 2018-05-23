#ifndef TIE_EXPR_HPP
#define TIE_EXPR_HPP

#include <mgcpp/expressions/generic_expr.hpp>

namespace mgcpp {

template <typename Expr>
struct tie_expr : public expression<Expr> {};

template <typename... Exprs>
struct symbolic_tie_expr : generic_expr<symbolic_tie_expr<Exprs...>,
                             0,
                             tie_expr,
                             std::tuple<typename Exprs::result_type...>,
                             0,
                             Exprs...> {
  using generic_expr<symbolic_tie_expr<Exprs...>,
                     0,
                     tie_expr,
                     std::tuple<typename Exprs::result_type...>,
                     0,
                     Exprs...>::generic_expr;
};

template <typename... Exprs>
inline symbolic_tie_expr<Exprs...> tie(Exprs const&... exprs);

}  // namespace mgcpp

#include <mgcpp/expressions/tie_expr.tpp>
#endif  // TIE_EXPR_HPP
