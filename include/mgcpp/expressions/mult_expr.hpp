
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

namespace mgcpp
{
    typename <typename LhsExpr, typename RhsExpr>
    struct mult_expr
    {
        RhsExpr& lhs; 
        LhsExpr& rhs;

        mult_expr(LhsExpr& lhs, RhsExpr& rhs);
    };

    inline mult_expr<typename LhsExpr, typename RhsExpr>
    operator*(LhsExpr& lhs, RhsExpr& rhs);
}
