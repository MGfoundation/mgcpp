
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DVEC_ELEMWISE_HPP_
#define _MGCPP_EXPRESSIONS_DVEC_ELEMWISE_HPP_

#include <mgcpp/expressions/dvec_expr.hpp>

namespace mgcpp
{
template <
    typename Expr,
    typename Expr::result_type (*Function)(
        typename Expr::result_type::parent_type const& vec)>
struct dvec_elemwise_expr : public dvec_expr<dvec_elemwise_expr<Expr, Function>>
{
    using expr_type = typename std::decay<Expr>::type;

    using result_type = typename expr_type::result_type;

    Expr _expr;

    inline explicit dvec_elemwise_expr(Expr const& expr) noexcept;
    inline explicit dvec_elemwise_expr(Expr&& expr) noexcept;

    inline decltype(auto) eval() const;
};

template <typename Expr>
inline decltype(auto) abs(dvec_expr<Expr> const& expr) noexcept;

template <typename Expr>
inline decltype(auto) sin(dvec_expr<Expr> const& expr) noexcept;

template <typename Expr>
inline decltype(auto) cos(dvec_expr<Expr> const& expr) noexcept;

template <typename Expr>
inline decltype(auto) tan(dvec_expr<Expr> const& expr) noexcept;

template <typename Expr>
inline decltype(auto) sinh(dvec_expr<Expr> const& expr) noexcept;

template <typename Expr>
inline decltype(auto) cosh(dvec_expr<Expr> const& expr) noexcept;

template <typename Expr>
inline decltype(auto) tanh(dvec_expr<Expr> const& expr) noexcept;

template <typename Expr>
inline decltype(auto) relu(dvec_expr<Expr> const& expr) noexcept;
} // namespace mgcpp

#endif
