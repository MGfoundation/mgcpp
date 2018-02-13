
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

namespace mgcpp
{
    template<typename LhsExpr, typename RhsExpr>
    dmat_dmat_add_expr<LhsExpr, RhsExpr> 
    operator-(dmat_expr<LhsExpr> const& lhs,
              dmat_expr<RhsExpr> const& rhs) noexcept
    {
        auto lhs_orig = ~lhs;
        auto rhs_orig = (-1) * (~rhs);
        return dmat_dmat_add_expr<LhsExpr, RhsExpr>(lhs_orig, rhs_orig);
    }

    template<typename LhsExpr, typename RhsExpr>
    dmat_dmat_add_expr<LhsExpr, RhsExpr> 
    sub(dmat_expr<LhsExpr> const& lhs,
        dmat_expr<RhsExpr> const& rhs) noexcept
    {
        auto lhs_orig = ~lhs;
        auto rhs_orig = (-1) * (~rhs);
        return dmat_dmat_add_expr<LhsExpr, RhsExpr>(lhs_orig, rhs_orig);
    }
}
