
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_SCALAR_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_SCALAR_EXPR_HPP_

#include <mgcpp/expressions/eval_context.hpp>
#include <mgcpp/expressions/expression.hpp>
#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/type_traits/is_scalar.hpp>
#include <type_traits>

namespace mgcpp {
template <typename Expr>
struct scalar_expr : public expression<Expr> {};

struct scalar_constant_expr_type;

template <typename T>
using scalar_constant_expr =
    generic_expr<scalar_constant_expr_type, 0, scalar_expr, T, 1, T>;

template <typename Type>
inline scalar_constant_expr<Type> scal(Type scalar);

struct scalar_zero_constant_expr_type;

template <typename T>
using scalar_zero_constant_expr =
    generic_expr<scalar_zero_constant_expr_type, 0, scalar_expr, T, 0>;

struct scalar_one_constant_expr_type;

template <typename T>
using scalar_one_constant_expr =
    generic_expr<scalar_one_constant_expr_type, 0, scalar_expr, T, 0>;

}  // namespace mgcpp

#include <mgcpp/expressions/scalar_expr.tpp>
#endif
