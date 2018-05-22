
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/scalar_dvec_mult.hpp>

#include <mgcpp/expressions/constant_expr.hpp>

namespace mgcpp {

namespace internal {
template <typename ScalExpr, typename DVecExpr>
auto scalar_dvec_mult_impl(scalar_expr<ScalExpr> const& scalar,
                           dvec_expr<DVecExpr> const& vec_expr) {
  return scalar_dvec_mult_expr<ScalExpr, DVecExpr>(~scalar, ~vec_expr);
}

// 0 * vector = zero vector
template <typename T, typename DVecExpr>
auto scalar_dvec_mult_impl(scalar_zero_constant_expr<T>,
                           dvec_expr<DVecExpr> const& vec_expr) {
  return make_zeros_like(~vec_expr);
}

// 1 * vector = the same vector
template <typename T, typename DVecExpr>
auto scalar_dvec_mult_impl(scalar_one_constant_expr<T>,
                           dvec_expr<DVecExpr> const& vec_expr) {
  return ~vec_expr;
}
}  // namespace internal

template <typename Scalar, typename DVecExpr, typename>
auto operator*(Scalar const& scalar,
               dvec_expr<DVecExpr> const& vec_expr) noexcept {
  auto scal_expr = mgcpp::scal(scalar);
  return internal::scalar_dvec_mult_impl(~scal_expr, ~vec_expr);
}

template <typename Scalar, typename DVecExpr, typename>
auto operator*(dvec_expr<DVecExpr> const& vec_expr,
               Scalar const& scalar) noexcept {
  auto scal_expr = mgcpp::scal(scalar);
  return internal::scalar_dvec_mult_impl(~scal_expr, ~vec_expr);
}

template <typename ScalExpr, typename DVecExpr>
auto operator*(scalar_expr<ScalExpr> const& scal_expr,
               dvec_expr<DVecExpr> const& vec_expr) noexcept {
  return internal::scalar_dvec_mult_impl(~scal_expr, ~vec_expr);
}

template <typename ScalExpr, typename DVecExpr>
auto operator*(dvec_expr<DVecExpr> const& vec_expr,
               scalar_expr<ScalExpr> const& scal_expr) noexcept {
  return internal::scalar_dvec_mult_impl(~scal_expr, ~vec_expr);
}

template <typename Scalar, typename DVecExpr, typename>
auto mult(Scalar const& scalar, dvec_expr<DVecExpr> const& vec_expr) noexcept {
  auto scal_expr = mgcpp::scal(scalar);
  return internal::scalar_dvec_mult_impl(~scal_expr, ~vec_expr);
}

template <typename Scalar, typename DVecExpr, typename>
auto mult(dvec_expr<DVecExpr> const& vec_expr, Scalar const& scalar) noexcept {
  auto scal_expr = mgcpp::scal(scalar);
  return internal::scalar_dvec_mult_impl(~scal_expr, ~vec_expr);
}

template <typename ScalExpr, typename DVecExpr>
auto mult(scalar_expr<ScalExpr> const& scal_expr,
          dvec_expr<DVecExpr> const& vec_expr) noexcept {
  return internal::scalar_dvec_mult_impl(~scal_expr, ~vec_expr);
}

template <typename ScalExpr, typename DVecExpr>
auto mult(dvec_expr<DVecExpr> const& vec_expr,
          scalar_expr<ScalExpr> const& scal_expr) noexcept {
  return internal::scalar_dvec_mult_impl(~scal_expr, ~vec_expr);
}
}  // namespace mgcpp
