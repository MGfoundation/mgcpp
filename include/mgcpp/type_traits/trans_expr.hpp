
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_TRANS_EXPR_HPP_
#define _MGCPP_TYPE_TRAITS_TRANS_EXPR_HPP_

#include <type_traits>

#include <mgcpp/gpu/forward.hpp>
#include <mgcpp/system/concept.hpp>
#include <mgcpp/type_traits/recursive_eval.hpp>
#include <mgcpp/type_traits/gpu_mat.hpp>

namespace mgcpp
{
    template<typename Expr>
    struct is_mat_trans_expr : std::false_type {};

    template<typename Expr, typename GpuMat,
             MGCPP_CONCEPT(assert_gpu_matrix<GpuMat>)>
    struct is_mat_trans_expr<trans<GpuMat>> : std::true_type {};

    template<typename... VariadicExpr>
    struct assert_mat_trans_expr
    {
        using result =
            typename std::enable_if <
            fold_or<is_mat_trans_expr, VariadicExpr...>::value
            >::type;
    };

    template<typename T, typename... VariadicExpr>
    struct assert_mat_trans_expr_t
    {
        using result =
            typename std::enable_if <
            fold_or<is_mat_trans_expr, VariadicExpr...>::value, T
            >::type;
    };

    template<typename Expr>
    struct assert_mat_trans_expr
    {
        using result =
            typename std::enable_if<
            is_mat_trans_expr<Expr>::value>>::type;
    };

}

#endif
