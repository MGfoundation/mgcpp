
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dvec_map.hpp>
#include <mgcpp/operations/map.hpp>

namespace mgcpp {

template <typename Expr>
inline decltype(auto) abs(dvec_expr<Expr> const& expr) noexcept {
  return dvec_map_expr<Expr, strict::abs>(~expr);
}

template <typename Expr>
inline decltype(auto) sin(dvec_expr<Expr> const& expr) noexcept {
  return dvec_map_expr<Expr, strict::sin>(~expr);
}

template <typename Expr>
inline decltype(auto) cos(dvec_expr<Expr> const& expr) noexcept {
  return dvec_map_expr<Expr, strict::cos>(~expr);
}

template <typename Expr>
inline decltype(auto) tan(dvec_expr<Expr> const& expr) noexcept {
  return dvec_map_expr<Expr, strict::tan>(~expr);
}

template <typename Expr>
inline decltype(auto) sinh(dvec_expr<Expr> const& expr) noexcept {
  return dvec_map_expr<Expr, strict::sinh>(~expr);
}

template <typename Expr>
inline decltype(auto) cosh(dvec_expr<Expr> const& expr) noexcept {
  return dvec_map_expr<Expr, strict::cosh>(~expr);
}

template <typename Expr>
inline decltype(auto) tanh(dvec_expr<Expr> const& expr) noexcept {
  return dvec_map_expr<Expr, strict::tanh>(~expr);
}

template <typename Expr>
inline decltype(auto) relu(dvec_expr<Expr> const& expr) noexcept {
  return dvec_map_expr<Expr, strict::relu>(~expr);
}
}  // namespace mgcpp
