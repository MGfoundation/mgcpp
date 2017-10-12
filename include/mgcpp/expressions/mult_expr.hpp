
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_MULT_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_MULT_EXPR_HPP_

namespace mgcpp
{
    template<typename LhsExpr, typename RhsExpr>
    struct mult_expr
    {
        RhsExpr&& _lhs; 
        LhsExpr&& _rhs;

        mult_expr(LhsExpr&& lhs, RhsExpr&& rhs);
    };

    template<typename LhsExpr, typename RhsExpr>
    inline mult_expr<LhsExpr, RhsExpr>
    operator*(LhsExpr&& lhs, RhsExpr&& rhs);
}

#endif
