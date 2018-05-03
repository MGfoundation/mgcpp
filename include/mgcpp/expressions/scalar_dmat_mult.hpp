
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_SCALAR_DMAT_MULT_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_SCALAR_DMAT_MULT_EXPR_HPP_

#include <type_traits>

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/generic_op.hpp>
#include <mgcpp/expressions/scalar_expr.hpp>

namespace mgcpp {

template <typename ScalExpr, typename DMatExpr>
using scalar_dmat_mult_expr = generic_op<expression_type::SCALAR_DMAT_MULT,
                                         dmat_expr,
                                         typename DMatExpr::result_type,
                                         ScalExpr,
                                         DMatExpr>;

/** Returns a scalar, dense matrix product expression.
 * \param lhs the left-hand side scalar variable
 * \param rhs the right-hand side dense matrix
 */
template <typename Scalar,
          typename DMatExpr,
          typename = typename std::enable_if<is_scalar<Scalar>::value>::type>
inline scalar_dmat_mult_expr<Scalar, DMatExpr> operator*(
    Scalar const& scalar,
    dmat_expr<DMatExpr> const& exp) noexcept;

/** Returns a scalar, dense matrix product expression.
 * \param lhs the left-hand side dense matrix
 * \param rhs the right-hand side scalar variable
 */
template <typename Scalar,
          typename DMatExpr,
          typename = typename std::enable_if<is_scalar<Scalar>::value>::type>
inline scalar_dmat_mult_expr<Scalar, DMatExpr> operator*(
    dmat_expr<DMatExpr> const& exp,
    Scalar const& scalar) noexcept;

/** Returns a scalar, dense matrix product expression.
 * \param lhs the left-hand side scalar expression
 * \param rhs the right-hand side dense matrix
 */
template <typename ScalExpr, typename DMatExpr>
inline scalar_dmat_mult_expr<ScalExpr, DMatExpr> operator*(
    scalar_expr<ScalExpr> const& scalar,
    dmat_expr<DMatExpr> const& exp) noexcept;

/** Returns a scalar, dense matrix product expression.
 * \param lhs the left-hand side dense matrix
 * \param rhs the right-hand side scalar expression
 */
template <typename ScalExpr, typename DMatExpr>
inline scalar_dmat_mult_expr<ScalExpr, DMatExpr> operator*(
    dmat_expr<DMatExpr> const& exp,
    scalar_expr<ScalExpr> const& scalar) noexcept;

/** Returns a scalar, dense matrix product expression.
 * \param lhs the left-hand side scalar variable
 * \param rhs the right-hand side dense matrix
 */
template <typename Scalar,
          typename DMatExpr,
          typename = typename std::enable_if<is_scalar<Scalar>::value>::type>
inline scalar_dmat_mult_expr<Scalar, DMatExpr> mult(
    Scalar const& scalar,
    dmat_expr<DMatExpr> const& mat_exp) noexcept;

/** Returns a scalar, dense matrix product expression.
 * \param lhs the left-hand side dense matrix
 * \param rhs the right-hand side scalar variable
 */
template <typename Scalar,
          typename DMatExpr,
          typename = typename std::enable_if<is_scalar<Scalar>::value>::type>
inline scalar_dmat_mult_expr<Scalar, DMatExpr> mult(
    dmat_expr<DMatExpr> const& mat_exp,
    Scalar const& scalar) noexcept;

/** Returns a scalar, dense matrix product expression.
 * \param lhs the left-hand side scalar expression
 * \param rhs the right-hand side dense matrix
 */
template <typename ScalExpr, typename DMatExpr>
inline scalar_dmat_mult_expr<ScalExpr, DMatExpr> mult(
    scalar_expr<ScalExpr> const& scalar,
    dmat_expr<DMatExpr> const& mat_exp) noexcept;

/** Returns a scalar, dense matrix product expression.
 * \param lhs the left-hand side dense matrix
 * \param rhs the right-hand side expression
 */
template <typename ScalExpr, typename DMatExpr>
inline scalar_dmat_mult_expr<ScalExpr, DMatExpr> mult(
    dmat_expr<DMatExpr> const& mat_exp,
    scalar_expr<ScalExpr> const& scalar) noexcept;
}  // namespace mgcpp

#endif
