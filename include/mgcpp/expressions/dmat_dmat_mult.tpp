
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>

#include <mgcpp/expressions/dmat_dmat_mult.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/system/assert.hpp>

namespace mgcpp
{
    template<typename LhsExpr, typename RhsExpr>
    dmat_dmat_mult_expr<LhsExpr, RhsExpr>::
    dmat_dmat_mult_expr(LhsExpr const& lhs, RhsExpr const& rhs) noexcept
        : _lhs(lhs),
          _rhs(rhs) {}

    template<typename LhsExpr, typename RhsExpr>
    typename dmat_dmat_mult_expr<LhsExpr, RhsExpr>::result_type
    dmat_dmat_mult_expr<LhsExpr, RhsExpr>::
    eval() const
    {
        auto lhs = mgcpp::eval(_lhs);
        auto rhs = mgcpp::eval(_rhs);

        MGCPP_ASSERT(lhs.shape()[1] == rhs.shape()[0],
                     "dimension doesn't match ");

        return strict::mult(lhs, rhs);
    }

    template<typename LhsExpr, typename RhsExpr>
    typename dmat_dmat_mult_expr<LhsExpr, RhsExpr>::result_type
    eval(dmat_dmat_mult_expr<LhsExpr, RhsExpr>&& expr)
    { expr.eval(); }

    template<typename LhsExpr, typename RhsExpr>
    dmat_dmat_mult_expr<LhsExpr, RhsExpr> 
    operator*(dmat_expr<LhsExpr> const& lhs,
              dmat_expr<RhsExpr> const& rhs) noexcept
    {
        return dmat_dmat_mult_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
    }

    template<typename LhsExpr, typename RhsExpr>
    dmat_dmat_mult_expr<LhsExpr, RhsExpr> 
    mult(dmat_expr<LhsExpr> const& lhs,
         dmat_expr<RhsExpr> const& rhs) noexcept
    {
        return dmat_dmat_mult_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
    }
}
