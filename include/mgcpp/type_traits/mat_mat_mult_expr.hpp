
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_MAT_MAT_MULT_EXPR_HPP_
#define _MGCPP_TYPE_TRAITS_MAT_MAT_MULT_EXPR_HPP_

#include <mgcpp/expressions/mat_mat_mult_expr.hpp>
#include <type_traits>

namespace mgcpp {
template <typename T>
struct is_mat_mat_mult_expr : std::false_type {};

template <typename LhsExpr, typename RhsExpr>
struct is_mat_mat_mult_expr<mat_mat_mult_expr<LhsExpr, RhsExpr>>
    : std::true_type {};
}  // namespace mgcpp

#endif
