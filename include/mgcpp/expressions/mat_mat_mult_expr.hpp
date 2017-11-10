
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_MAT_MAT_MULT_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_MAT_MAT_MULT_EXPR_HPP_

#include <mgcpp/device/forward.hpp>
#include <mgcpp/expressions/expr_eval.hpp>
#include <mgcpp/expressions/mat_expr.hpp>
#include <mgcpp/type_traits/mat_expr.hpp>
#include <mgcpp/type_traits/type_traits.hpp>

namespace mgcpp
{
    template<typename LhsExpr, typename RhsExpr>
    struct mat_mat_mult_expr_t { };

    template<typename LhsExpr, typename RhsExpr>
    struct mat_expr<mat_mat_mult_expr_t<LhsExpr, RhsExpr>>
    {
        using lhs_expr_type = typename std::decay<LhsExpr>::type;
        using rhs_expr_type = typename std::decay<RhsExpr>::type;

        using result_type = typename lhs_expr_type::result_type;

        LhsExpr&& _lhs;
        RhsExpr&& _rhs;

        inline mat_expr(LhsExpr&& lhs, RhsExpr&& rhs) noexcept;

        inline result_type
        eval() const;
    };

    template<typename LhsExpr, typename RhsExpr>
    using mat_mat_mult_expr = mat_expr<mat_mat_mult_expr_t<LhsExpr, RhsExpr>>;

    template<typename LhsExpr, typename RhsExpr>
    inline typename mat_mat_mult_expr<LhsExpr, RhsExpr>::result_type
    eval(mat_mat_mult_expr<LhsExpr, RhsExpr> const& expr);

    template<typename LhsExpr, typename RhsExpr,
             MGCPP_CONCEPT(is_mat_expr<LhsExpr>::value
                           && is_mat_expr<RhsExpr>::value)>
    inline mat_mat_mult_expr<LhsExpr, RhsExpr> 
    operator*(LhsExpr&& lhs, RhsExpr&& rhs) noexcept;
}

#include <mgcpp/expressions/mat_mat_mult_expr.tpp>
#endif
