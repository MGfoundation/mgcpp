
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DVEC_ELEMWISE_HPP_
#define _MGCPP_EXPRESSIONS_DVEC_ELEMWISE_HPP_

#include <mgcpp/expressions/dvec_expr.hpp>
#include <mgcpp/expressions/expr_eval.hpp>
#include <mgcpp/vector/device_vector.hpp>

namespace mgcpp {
template <typename Expr,
          typename VectorType,
          VectorType (*Function)(typename VectorType::parent_type const& vec)>
struct dvec_elemwise_expr
    : public dvec_expr<dvec_elemwise_expr<Expr, VectorType, Function>> {
  using expr_type = typename std::decay<Expr>::type;

  using result_type = typename expr_type::result_type;

  Expr const& _expr;

  inline dvec_elemwise_expr(Expr const& expr) noexcept;

  inline decltype(auto) eval() const;
};

template <typename Expr,
          typename VectorType,
          VectorType (*Function)(typename VectorType::parent_type const& vec)>
inline decltype(auto) eval(
    dvec_elemwise_expr<Expr, VectorType, Function> const& expr);

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
}  // namespace mgcpp

#include <mgcpp/expressions/dvec_elemwise.tpp>
#endif
