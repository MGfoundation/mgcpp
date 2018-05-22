
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/scalar_dmat_mult.hpp>

#include <mgcpp/expressions/constant_expr.hpp>

namespace mgcpp {

namespace internal {
template <typename ScalExpr, typename DMatExpr>
auto scalar_dmat_mult_impl(scalar_expr<ScalExpr> const& scalar,
                           dmat_expr<DMatExpr> const& mat_expr) {
  return scalar_dmat_mult_expr<ScalExpr, DMatExpr>(~scalar, ~mat_expr);
}

// 0 * matrix = zero matrix
template <typename T, typename DMatExpr>
auto scalar_dmat_mult_impl(scalar_zero_constant_expr<T>,
                           dmat_expr<DMatExpr> const& mat_expr) {
  return make_zeros_like(~mat_expr);
}

// 1 * matrix = the same matrix
template <typename T, typename DMatExpr>
auto scalar_dmat_mult_impl(scalar_one_constant_expr<T>,
                           dmat_expr<DMatExpr> const& mat_expr) {
  return ~mat_expr;
}
}  // namespace internal

template <typename Scalar, typename DMatExpr, typename>
auto operator*(Scalar const& scalar,
               dmat_expr<DMatExpr> const& mat_expr) noexcept {
  auto scal_expr = mgcpp::scal(scalar);
  return internal::scalar_dmat_mult_impl(~scal_expr, ~mat_expr);
}

template <typename Scalar, typename DMatExpr, typename>
auto operator*(dmat_expr<DMatExpr> const& mat_expr,
               Scalar const& scalar) noexcept {
  auto scal_expr = mgcpp::scal(scalar);
  return internal::scalar_dmat_mult_impl(~scal_expr, ~mat_expr);
}

template <typename ScalExpr, typename DMatExpr>
auto operator*(scalar_expr<ScalExpr> const& scal_expr,
               dmat_expr<DMatExpr> const& mat_expr) noexcept {
  return internal::scalar_dmat_mult_impl(~scal_expr, ~mat_expr);
}

template <typename ScalExpr, typename DMatExpr>
auto operator*(dmat_expr<DMatExpr> const& mat_expr,
               scalar_expr<ScalExpr> const& scal_expr) noexcept {
  return internal::scalar_dmat_mult_impl(~scal_expr, ~mat_expr);
}

template <typename Scalar, typename DMatExpr, typename>
auto mult(Scalar const& scalar, dmat_expr<DMatExpr> const& mat_expr) noexcept {
  auto scal_expr = mgcpp::scal(scalar);
  return internal::scalar_dmat_mult_impl(~scal_expr, ~mat_expr);
}

template <typename Scalar, typename DMatExpr, typename>
auto mult(dmat_expr<DMatExpr> const& mat_expr, Scalar const& scalar) noexcept {
  auto scal_expr = mgcpp::scal(scalar);
  return internal::scalar_dmat_mult_impl(~scal_expr, ~mat_expr);
}

template <typename ScalExpr, typename DMatExpr>
auto mult(scalar_expr<ScalExpr> const& scal_expr,
          dmat_expr<DMatExpr> const& mat_expr) noexcept {
  return internal::scalar_dmat_mult_impl(~scal_expr, ~mat_expr);
}

template <typename ScalExpr, typename DMatExpr>
auto mult(dmat_expr<DMatExpr> const& mat_expr,
          scalar_expr<ScalExpr> const& scal_expr) noexcept {
  return internal::scalar_dmat_mult_impl(~scal_expr, ~mat_expr);
}
}  // namespace mgcpp
