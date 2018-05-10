
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/scalar_dmat_mult.hpp>
#include <mgcpp/operations/mult.hpp>

namespace mgcpp {

template <typename Scalar, typename DMatExpr, typename>
scalar_dmat_mult_expr<scalar_constant_expr<Scalar>, DMatExpr> operator*(
    Scalar const& scalar,
    dmat_expr<DMatExpr> const& mat_expr) noexcept {
  auto scal_expr = mgcpp::scal(scalar);
  return scalar_dmat_mult_expr<decltype(scal_expr), DMatExpr>(scal_expr, ~mat_expr);
}

template <typename Scalar, typename DMatExpr, typename>
scalar_dmat_mult_expr<scalar_constant_expr<Scalar>, DMatExpr> operator*(
    dmat_expr<DMatExpr> const& mat_expr,
    Scalar const& scalar) noexcept {
  auto scal_expr = mgcpp::scal(scalar);
  return scalar_dmat_mult_expr<decltype(scal_expr), DMatExpr>(scal_expr, ~mat_expr);
}

template <typename ScalExpr, typename DMatExpr>
scalar_dmat_mult_expr<ScalExpr, DMatExpr> operator*(
    scalar_expr<ScalExpr> const& scalar,
    dmat_expr<DMatExpr> const& mat_expr) noexcept {
  return scalar_dmat_mult_expr<ScalExpr, DMatExpr>(~scalar, ~mat_expr);
}

template <typename ScalExpr, typename DMatExpr>
scalar_dmat_mult_expr<ScalExpr, DMatExpr> operator*(
    dmat_expr<DMatExpr> const& mat_expr,
    scalar_expr<ScalExpr> const& scalar) noexcept {
  return scalar_dmat_mult_expr<ScalExpr, DMatExpr>(~scalar, ~mat_expr);
}

template <typename Scalar, typename DMatExpr, typename>
scalar_dmat_mult_expr<scalar_constant_expr<Scalar>, DMatExpr> mult(
    Scalar const& scalar,
    dmat_expr<DMatExpr> const& mat_expr) noexcept {
  auto scal_expr = mgcpp::scal(scalar);
  return scalar_dmat_mult_expr<decltype(scal_expr), DMatExpr>(scal_expr, ~mat_expr);
}

template <typename Scalar, typename DMatExpr, typename>
scalar_dmat_mult_expr<scalar_constant_expr<Scalar>, DMatExpr> mult(
    dmat_expr<DMatExpr> const& mat_expr,
    Scalar const& scalar) noexcept {
  auto scal_expr = mgcpp::scal(scalar);
  return scalar_dmat_mult_expr<decltype(scal_expr), DMatExpr>(scal_expr, ~mat_expr);
}

template <typename ScalExpr, typename DMatExpr>
scalar_dmat_mult_expr<ScalExpr, DMatExpr> mult(
    scalar_expr<ScalExpr> const& scalar,
    dmat_expr<DMatExpr> const& mat_expr) noexcept {
  return scalar_dmat_mult_expr<ScalExpr, DMatExpr>(~scalar, ~mat_expr);
}

template <typename ScalExpr, typename DMatExpr>
scalar_dmat_mult_expr<ScalExpr, DMatExpr> mult(
    dmat_expr<DMatExpr> const& mat_expr,
    scalar_expr<ScalExpr> const& scalar) noexcept {
  return scalar_dmat_mult_expr<ScalExpr, DMatExpr>(~scalar, ~mat_expr);
}
}  // namespace mgcpp
