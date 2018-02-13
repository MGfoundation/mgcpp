
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dmat_dmat_add.hpp>
#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/expr_eval.hpp>
#include <mgcpp/matrix/device_matrix.hpp>

#ifndef _MGCPP_EXPRESSIONS_DMAT_DMAT_SUB_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_DMAT_SUB_EXPR_HPP_

namespace mgcpp
{
    template<typename LhsExpr, typename RhsExpr>
    inline dmat_dmat_add_expr<LhsExpr, RhsExpr> 
    operator-(dmat_expr<LhsExpr> const& lhs,
              dmat_expr<RhsExpr> const& rhs) noexcept;

    template<typename LhsExpr, typename RhsExpr>
    inline dmat_dmat_add_expr<LhsExpr, RhsExpr> 
    sub(dmat_expr<LhsExpr> const& lhs,
        dmat_expr<RhsExpr> const& rhs) noexcept;
    
}

#include <mgcpp/expressions/dmat_dmat_sub.tpp>
#endif
