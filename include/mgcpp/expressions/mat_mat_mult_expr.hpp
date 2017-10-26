
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_MAT_MAT_MULT_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_MAT_MAT_MULT_EXPR_HPP_

#include <mgcpp/device/forward.hpp>
#include <mgcpp/expressions/expr_eval.hpp>
#include <mgcpp/expressions/mat_expr.hpp>
#include <mgcpp/expressions/result_type.hpp>
#include <mgcpp/system/concept.hpp>
#include <mgcpp/type_traits/mat_expr.hpp>
#include <mgcpp/type_traits/type_traits.hpp>

namespace mgcpp
{
    template<typename LhsExpr, typename RhsExpr>
    struct mat_mat_mult_expr
    {
        using result_type =
            typename result_type<LhsExpr>::type;

        LhsExpr&& _lhs;
        RhsExpr&& _rhs;

        inline mat_mat_mult_expr(LhsExpr&& lhs,
                                 RhsExpr&& rhs) noexcept;

        inline result_type
        eval();
    };

    template<typename LhsExpr, typename RhsExpr>
    struct result_type<
        mat_mat_mult_expr<LhsExpr, RhsExpr>,
        typename assert_both_mat_expr<LhsExpr, RhsExpr>::type>
    {
        using type =
            typename mat_mat_mult_expr<LhsExpr, RhsExpr>::result_type;
    };

    template<typename LhsExpr, typename RhsExpr>
    inline typename mat_mat_mult_expr<LhsExpr, RhsExpr>::result_type
    eval(mat_mat_mult_expr<LhsExpr, RhsExpr>& expr);

    template<typename LhsExpr, typename RhsExpr,
             typename = typename
             assert_both_mat_expr<LhsExpr, RhsExpr>::result>
    inline mat_mat_mult_expr<LhsExpr, RhsExpr> 
    operator*(LhsExpr&& lhs, RhsExpr&& rhs) noexcept;
}

#include <mgcpp/expressions/mat_mat_mult_expr.tpp>
#endif
