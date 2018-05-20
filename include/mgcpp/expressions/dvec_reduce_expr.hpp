
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DVEC_REDUCE_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DVEC_REDUCE_EXPR_HPP_

#include <mgcpp/expressions/dvec_expr.hpp>
#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/expressions/scalar_expr.hpp>

namespace mgcpp {

struct dvec_reduce_sum_expr_type;

template <typename Expr>
using dvec_reduce_sum_expr =
    generic_expr<dvec_reduce_sum_expr_type,
                 0,
                 scalar_expr,
                 typename Expr::result_type::value_type,
                 0,
                 Expr>;

template <typename Expr>
inline decltype(auto) reduce_sum(dvec_expr<Expr> const& expr) noexcept;

struct dvec_reduce_mean_expr_type;

template <typename Expr>
using dvec_reduce_mean_expr =
    generic_expr<dvec_reduce_mean_expr_type,
                 0,
                 scalar_expr,
                 typename Expr::result_type::value_type,
                 0,
                 Expr>;

template <typename Expr>
inline decltype(auto) reduce_mean(dvec_expr<Expr> const& expr) noexcept;
}  // namespace mgcpp

#include <mgcpp/expressions/dvec_reduce_expr.tpp>
#endif
