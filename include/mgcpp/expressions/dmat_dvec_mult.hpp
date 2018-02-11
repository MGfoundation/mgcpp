
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DMAT_DVEC_MULT_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_DVEC_MULT_EXPR_HPP_

#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/expressions/expr_eval.hpp>
#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/dvec_expr.hpp>

namespace mgcpp
{
    template<typename MatExpr, typename VecExpr>
    struct dmat_dvec_mult_expr
        : public dvec_expr<dmat_dvec_mult_expr<MatExpr, VecExpr>>
    {
        using lhs_expr_type = typename std::decay<MatExpr>::type;
        using rhs_expr_type = typename std::decay<VecExpr>::type;

        using result_type = typename rhs_expr_type::result_type;

        MatExpr const& _mat;
        VecExpr const& _vec;

        inline dmat_dvec_mult_expr(MatExpr const& mat,
                                   VecExpr const& vec) noexcept;

        inline result_type
        eval() const;
    };

    template<typename MatExpr, typename VecExpr>
    inline typename dmat_dvec_mult_expr<MatExpr, VecExpr>::result_type
    eval(dmat_dvec_mult_expr<MatExpr, VecExpr> const& expr);

    template<typename MatExpr, typename VecExpr>
    inline dmat_dvec_mult_expr<MatExpr, VecExpr>
    operator*(dmat_expr<MatExpr> const& mat,
              dvec_expr<VecExpr> const& vec) noexcept;

    template<typename MatExpr, typename VecExpr>
    inline dmat_dvec_mult_expr<MatExpr, VecExpr>
    mult(dmat_expr<MatExpr> const& mat,
              dvec_expr<VecExpr> const& vec) noexcept;
}


#include <mgcpp/expressions/dmat_dvec_mult.tpp>
#endif
