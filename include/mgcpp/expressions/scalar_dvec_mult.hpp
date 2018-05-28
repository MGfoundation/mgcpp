
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef MGCPP_EXPRESSIONS_SCALAR_DVEC_MULT_EXPR_HPP
#define MGCPP_EXPRESSIONS_SCALAR_DVEC_MULT_EXPR_HPP

#include <type_traits>

#include <mgcpp/expressions/dvec_expr.hpp>
#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/expressions/scalar_expr.hpp>

namespace mgcpp {

struct scalar_dvec_mult_expr_type;

template <typename ScalExpr, typename DVecExpr>
using scalar_dvec_mult_expr = binary_expr<scalar_dvec_mult_expr_type,
                                          dvec_expr,
                                          typename DVecExpr::result_type,
                                          ScalExpr,
                                          DVecExpr>;

/** Returns a scalar, dense vector product expression.
 * \param lhs the left-hand side scalar variable
 * \param rhs the right-hand side dense vector
 */
template <typename Scalar,
          typename DVecExpr,
          typename = typename std::enable_if<is_scalar<Scalar>::value>::type>
inline auto operator*(Scalar const& scalar,
                      dvec_expr<DVecExpr> const& exp) noexcept;

/** Returns a scalar, dense vector product expression.
 * \param lhs the left-hand side dense vector
 * \param rhs the right-hand side scalar variable
 */
template <typename Scalar,
          typename DVecExpr,
          typename = typename std::enable_if<is_scalar<Scalar>::value>::type>
inline auto operator*(dvec_expr<DVecExpr> const& exp,
                      Scalar const& scalar) noexcept;

/** Returns a scalar, dense vector product expression.
 * \param lhs the left-hand side scalar expression
 * \param rhs the right-hand side dense vector
 */
template <typename ScalExpr, typename DVecExpr>
inline auto operator*(scalar_expr<ScalExpr> const& scalar,
                      dvec_expr<DVecExpr> const& exp) noexcept;

/** Returns a scalar, dense vector product expression.
 * \param lhs the left-hand side dense vector
 * \param rhs the right-hand side scalar expression
 */
template <typename ScalExpr, typename DVecExpr>
inline auto operator*(dvec_expr<DVecExpr> const& exp,
                      scalar_expr<ScalExpr> const& scalar) noexcept;

/** Returns a scalar, dense vector product expression.
 * \param lhs the left-hand side scalar variable
 * \param rhs the right-hand side dense vector
 */
template <typename Scalar,
          typename DVecExpr,
          typename = typename std::enable_if<is_scalar<Scalar>::value>::type>
inline auto mult(Scalar const& scalar,
                 dvec_expr<DVecExpr> const& vec_exp) noexcept;

/** Returns a scalar, dense vector product expression.
 * \param lhs the left-hand side dense vector
 * \param rhs the right-hand side scalar variable
 */
template <typename Scalar,
          typename DVecExpr,
          typename = typename std::enable_if<is_scalar<Scalar>::value>::type>
inline auto mult(dvec_expr<DVecExpr> const& vec_exp,
                 Scalar const& scalar) noexcept;

/** Returns a scalar, dense vector product expression.
 * \param lhs the left-hand side scalar expression
 * \param rhs the right-hand side dense vector
 */
template <typename ScalExpr, typename DVecExpr>
inline auto mult(scalar_expr<ScalExpr> const& scalar,
                 dvec_expr<DVecExpr> const& vec_exp) noexcept;

/** Returns a scalar, dense vector product expression.
 * \param lhs the left-hand side dense vector
 * \param rhs the right-hand side expression
 */
template <typename ScalExpr, typename DVecExpr>
inline auto mult(dvec_expr<DVecExpr> const& vec_exp,
                 scalar_expr<ScalExpr> const& scalar) noexcept;
}  // namespace mgcpp

#include <mgcpp/expressions/scalar_dvec_mult.tpp>
#endif
