#ifndef TIE_EXPR_HPP
#define TIE_EXPR_HPP

#include <mgcpp/expressions/generic_expr.hpp>

namespace mgcpp {

template <typename Expr>
struct tie_expr : public expression<Expr> {};

template <typename... Exprs>
using tie_op = generic_expr<expression_type,
                          expression_type::TIE,
                          tie_expr,
                          std::tuple<typename Exprs::result_type...>,
                          0,
                          Exprs...>;

template <typename... Exprs>
inline tie_op<Exprs...> tie(Exprs const& ... exprs);

}

#include <mgcpp/expressions/tie_expr.tpp>
#endif // TIE_EXPR_HPP
