
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>

#include <mgcpp/expressions/mat_mat_mult_expr.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/system/assert.hpp>

namespace mgcpp
{
    template<typename LhsExpr, typename RhsExpr>
    mat_expr<mat_mat_mult_expr_t<LhsExpr, RhsExpr>>::
    mat_expr(LhsExpr&& lhs, RhsExpr&& rhs) noexcept
        : _lhs(std::forward<LhsExpr>(lhs)),
          _rhs(std::forward<RhsExpr>(rhs)) {}

    template<typename LhsExpr, typename RhsExpr>
    typename mat_expr<mat_mat_mult_expr_t<LhsExpr, RhsExpr>>::result_type
    mat_expr<mat_mat_mult_expr_t<LhsExpr, RhsExpr>>::
    eval() const
    {
        auto lhs = mgcpp::eval(_lhs);
        auto rhs = mgcpp::eval(_rhs);

        MGCPP_ASSERT(lhs.shape()[1] == rhs.shape()[0],
                     "dimension doesn't match ");

        return strict::mult(lhs, rhs);
    }

    template<typename LhsExpr, typename RhsExpr>
    typename mat_mat_mult_expr<LhsExpr, RhsExpr>::result_type
    eval(mat_mat_mult_expr<LhsExpr, RhsExpr>&& expr)
    {
        expr.eval();
    }

    template<typename LhsExpr, typename RhsExpr,
             typename>
    mat_mat_mult_expr<LhsExpr, RhsExpr> 
    operator*(LhsExpr&& lhs, RhsExpr&& rhs) noexcept
    {
        return mat_mat_mult_expr<LhsExpr, RhsExpr>(
            std::forward<LhsExpr>(lhs),
            std::forward<RhsExpr>(rhs));
    }
}
