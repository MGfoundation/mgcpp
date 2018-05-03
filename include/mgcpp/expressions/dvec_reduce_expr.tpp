
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dvec_reduce_expr.hpp>
#include <mgcpp/operations/sum.hpp>
#include <mgcpp/operations/mean.hpp>

namespace mgcpp
{
template <typename Expr>
decltype(auto) reduce_sum(const dvec_expr<Expr>& expr) noexcept
{
    return dvec_reduce_expr<Expr, strict::sum>(~expr);
}

template <typename Expr>
decltype(auto) reduce_mean(const dvec_expr<Expr>& expr) noexcept
{
    return dvec_reduce_expr<Expr, strict::mean>(~expr);
}
}
