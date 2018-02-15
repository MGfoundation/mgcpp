
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/scalar_dmat_mult.hpp>
#include <mgcpp/operations/mult.hpp>

namespace mgcpp {
template <typename ScalExpr, typename DMatExpr>
scalar_dmat_mult_expr<ScalExpr, DMatExpr>::scalar_dmat_mult_expr(
    ScalExpr const& scal_expr,
    DMatExpr const& dmat_expr) noexcept
    : _scal_expr(scal_expr), _dmat_expr(dmat_expr) {}

template <typename ScalExpr, typename DMatExpr>
typename scalar_dmat_mult_expr<ScalExpr, DMatExpr>::result_type
scalar_dmat_mult_expr<ScalExpr, DMatExpr>::eval(bool eval_trans) const {
  (void)eval_trans;

  auto const& scal = mgcpp::eval(_scal_expr);
  auto const& dmat = mgcpp::eval(_dmat_expr, false);

  return strict::mult(scal, dmat);
}

template <typename ScalExpr, typename DMatExpr>
typename scalar_dmat_mult_expr<ScalExpr, DMatExpr>::result_type eval(
    scalar_dmat_mult_expr<ScalExpr, DMatExpr> const& expr,
    bool eval_trans) {
  return expr.eval(eval_trans);
}

template <typename Scalar, typename DMatExpr, typename>
scalar_dmat_mult_expr<Scalar, DMatExpr> operator*(
    Scalar const& scalar,
    dmat_expr<DMatExpr> const& mat_expr) noexcept {
  return scalar_dmat_mult_expr<Scalar, DMatExpr>(scalar, ~mat_expr);
}

template <typename Scalar, typename DMatExpr, typename>
inline scalar_dmat_mult_expr<Scalar, DMatExpr> operator*(
    dmat_expr<DMatExpr> const& mat_expr,
    Scalar const& scalar) noexcept {
  return scalar_dmat_mult_expr<Scalar, DMatExpr>(scalar, ~mat_expr);
}

template <typename ScalExpr, typename DMatExpr>
inline scalar_dmat_mult_expr<ScalExpr, DMatExpr> operator*(
    scalar_expr<ScalExpr> const& scalar,
    dmat_expr<DMatExpr> const& mat_expr) noexcept {
  return scalar_dmat_mult_expr<ScalExpr, DMatExpr>(~scalar, ~mat_expr);
}

template <typename ScalExpr, typename DMatExpr>
inline scalar_dmat_mult_expr<ScalExpr, DMatExpr> operator*(
    dmat_expr<DMatExpr> const& mat_expr,
    scalar_expr<ScalExpr> const& scalar) noexcept {
  return scalar_dmat_mult_expr<ScalExpr, DMatExpr>(~scalar, ~mat_expr);
}

template <typename Scalar, typename DMatExpr, typename>
scalar_dmat_mult_expr<Scalar, DMatExpr> mult(
    Scalar const& scalar,
    dmat_expr<DMatExpr> const& mat_expr) noexcept {
  return scalar_dmat_mult_expr<Scalar, DMatExpr>(scalar, ~mat_expr);
}

template <typename Scalar, typename DMatExpr, typename>
inline scalar_dmat_mult_expr<Scalar, DMatExpr> mult(
    dmat_expr<DMatExpr> const& mat_expr,
    Scalar const& scalar) noexcept {
  return scalar_dmat_mult_expr<Scalar, DMatExpr>(scalar, ~mat_expr);
}

template <typename ScalExpr, typename DMatExpr>
inline scalar_dmat_mult_expr<ScalExpr, DMatExpr> mult(
    scalar_expr<ScalExpr> const& scalar,
    dmat_expr<DMatExpr> const& mat_expr) noexcept {
  return scalar_dmat_mult_expr<ScalExpr, DMatExpr>(~scalar, ~mat_expr);
}

template <typename ScalExpr, typename DMatExpr>
inline scalar_dmat_mult_expr<ScalExpr, DMatExpr> mult(
    dmat_expr<DMatExpr> const& mat_expr,
    scalar_expr<ScalExpr> const& scalar) noexcept {
  return scalar_dmat_mult_expr<ScalExpr, DMatExpr>(~scalar, ~mat_expr);
}
}  // namespace mgcpp
