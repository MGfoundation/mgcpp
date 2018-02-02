

//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>

#include <mgcpp/expressions/dmat_dmat_add.hpp>
#include <mgcpp/operations/add.hpp>
#include <mgcpp/operations/gemm.hpp>
#include <mgcpp/system/assert.hpp>

namespace mgcpp
{
    namespace internal
    {
        struct dmat_dmat_add_subgraph_matcher
        {
            template<typename LhsExpr, typename RhsExpr>
            static decltype(auto) eval(dmat_dmat_add_expr<LhsExpr, RhsExpr> const& expr)
            {
                auto lhs = mgcpp::eval(expr._lhs);
                auto rhs = mgcpp::eval(expr._rhs);

                return strict::add(lhs, rhs);
            }

            template<typename AType, typename BType, typename CType>
            static decltype(auto) eval(dmat_dmat_add_expr<dmat_dmat_mult_expr<AType, BType>, CType> const& expr)
            {
                auto A = mgcpp::eval(expr._lhs._lhs);
                auto B = mgcpp::eval(expr._lhs._rhs);
                auto C = mgcpp::eval(expr._rhs);
                return strict::gemm(A, B, C);
            }

            template<typename AType, typename BType, typename CType>
            static decltype(auto) eval(dmat_dmat_add_expr<CType, dmat_dmat_mult_expr<AType, BType>> const& expr)
            {
                auto A = mgcpp::eval(expr._rhs._lhs);
                auto B = mgcpp::eval(expr._rhs._rhs);
                auto C = mgcpp::eval(expr._lhs);
                return strict::gemm(A, B, C);
            }
        };
    }

    template<typename LhsExpr, typename RhsExpr>
    dmat_dmat_add_expr<LhsExpr, RhsExpr>::
    dmat_dmat_add_expr(LhsExpr const& lhs, RhsExpr const& rhs) noexcept
        : _lhs(lhs),
          _rhs(rhs) {}
    
    template<typename LhsExpr, typename RhsExpr>
    decltype(auto)
    dmat_dmat_add_expr<LhsExpr, RhsExpr>::
    eval() const
    {
        return internal::dmat_dmat_add_subgraph_matcher::eval(*this);
    }

    template<typename LhsExpr, typename RhsExpr>
    typename dmat_dmat_add_expr<LhsExpr, RhsExpr>::result_type
    eval(dmat_dmat_add_expr<LhsExpr, RhsExpr>&& expr)
    { expr.eval(); }

    template<typename LhsExpr, typename RhsExpr>
    dmat_dmat_add_expr<LhsExpr, RhsExpr> 
    operator+(dmat_expr<LhsExpr> const& lhs,
              dmat_expr<RhsExpr> const& rhs) noexcept
    {
        return dmat_dmat_add_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
    }

    template<typename LhsExpr, typename RhsExpr>
    dmat_dmat_add_expr<LhsExpr, RhsExpr> 
    add(dmat_expr<LhsExpr> const& lhs,
        dmat_expr<RhsExpr> const& rhs) noexcept
    {
        return dmat_dmat_add_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
    }
}
