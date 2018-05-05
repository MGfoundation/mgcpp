
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DMAT_TRANS_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DMAT_TRANS_EXPR_HPP_

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/matrix/forward.hpp>

namespace mgcpp {

template <typename Expr>
using dmat_trans_expr = unary_expr<expression_type::DMAT_TRANSPOSE,
                                 dmat_expr,
                                 typename Expr::result_type,
                                 Expr>;

template <typename Expr>
inline dmat_trans_expr<Expr> trans(dmat_expr<Expr> const& expr) noexcept;
}  // namespace mgcpp

#include <mgcpp/expressions/dmat_trans_expr.tpp>
#endif
