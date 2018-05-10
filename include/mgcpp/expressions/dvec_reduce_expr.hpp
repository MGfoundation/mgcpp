
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

template <typename Expr,
          typename Expr::result_type::value_type (*Function)(
              typename Expr::result_type::parent_type const& vec)>
using dvec_reduce_expr =
    generic_expr<typename Expr::result_type::value_type (*)(
                   typename Expr::result_type::parent_type const& vec),
               Function,
               scalar_expr,
               typename Expr::result_type::value_type,
               0,
               Expr>;

template <typename Expr>
inline decltype(auto) reduce_sum(dvec_expr<Expr> const& expr) noexcept;

template <typename Expr>
inline decltype(auto) reduce_mean(dvec_expr<Expr> const& expr) noexcept;
}  // namespace mgcpp

#include <mgcpp/expressions/dvec_reduce_expr.tpp>
#endif
