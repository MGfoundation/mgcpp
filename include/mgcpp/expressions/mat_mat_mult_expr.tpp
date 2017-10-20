
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>

#include <mgcpp/expressions/mat_mat_mult_expr.hpp>
#include <mgcpp/operations/mult.hpp>

namespace mgcpp
{
    template<typename LhsExpr, typename RhsExpr>
    mat_mat_mult_expr<LhsExpr, RhsExpr>::
    mat_mat_mult_expr(LhsExpr&& lhs, RhsExpr&& rhs) noexcept
        :_lhs(std::forward<LhsExpr>(lhs)),
         _rhs(std::forward<RhsExpr>(rhs)) {}

    template<typename LhsExpr, typename RhsExpr>
    typename mat_mat_mult_expr<LhsExpr, RhsExpr>::result_type
    mat_mat_mult_expr<LhsExpr, RhsExpr>::
    eval()
    {
        using first_type =
            typename mgcpp::result_type<LhsExpr>::type;
        using second_type =
            typename mgcpp::result_type<RhsExpr>::type;

        if(!std::is_same<first_type, second_type>::value)
            MGCPP_THROW_RUNTIME_ERROR("type of matrices not equal");

        auto lhs = mgcpp::eval(_lhs);
        auto rhs = mgcpp::eval(_rhs);

        auto lhs_shape = lhs.shape();
        auto rhs_shape = rhs.shape();

        if(lhs_shape.second != rhs_shape.first)
            MGCPP_THROW_LENGTH_ERROR("dimension doesn't match");

        return strict::mult(lhs, rhs);
    }

    template<typename LhsExpr, typename RhsExpr>
    inline typename result_type<
        mat_mat_mult_expr<LhsExpr, RhsExpr>>::type
    eval(mat_mat_mult_expr<LhsExpr, RhsExpr>&& expr)
    {
        expr.eval();
    }

    template<typename LhsExpr, typename RhsExpr,
             typename>
    mat_mat_mult_expr<LhsExpr, RhsExpr> 
    gpu::
    operator*(LhsExpr&& lhs, RhsExpr&& rhs) noexcept
    {
        return mat_mat_mult_expr<LhsExpr, RhsExpr>(
            std::forward<LhsExpr>(lhs),
            std::forward<RhsExpr>(rhs));
    }
}
