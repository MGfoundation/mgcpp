
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// This file contains all the forward-declarations of the expressions.
// A *.tpp file in the expressions directory should include this file.

#ifndef _MGCPP_EXPRESSIONS_FORWARD_HPP_
#define _MGCPP_EXPRESSIONS_FORWARD_HPP_

#include <mgcpp/matrix/forward.hpp>
#include <mgcpp/type_traits/type_traits.hpp>
#include <mgcpp/vector/forward.hpp>
#include <type_traits>

namespace mgcpp {

template <typename Type>
struct expression;

template <typename Expr>
struct dmat_expr;

template <typename Expr>
struct dvec_expr;

template <typename DenseMatrix, typename Type, size_t DeviceId>
decltype(auto) eval(
    dense_vector<DenseMatrix, Type, DeviceId> const& device_vec);

template <typename LhsExpr, typename RhsExpr>
struct dmat_dmat_add_expr;

template <typename LhsExpr, typename RhsExpr>
inline typename dmat_dmat_add_expr<LhsExpr, RhsExpr>::result_type eval(
    dmat_dmat_add_expr<LhsExpr, RhsExpr> const& expr,
    bool eval_trans = true);

template <typename LhsExpr, typename RhsExpr>
struct dmat_dmat_mult_expr;

template <typename LhsExpr, typename RhsExpr>
inline typename dmat_dmat_mult_expr<LhsExpr, RhsExpr>::result_type eval(
    dmat_dmat_mult_expr<LhsExpr, RhsExpr> const& expr,
    bool eval_trans = true);

template <typename ScalExpr, typename DMatExpr>
struct scalar_dmat_mult_expr;

template <typename ScalExpr, typename DMatExpr>
inline typename scalar_dmat_mult_expr<ScalExpr, DMatExpr>::result_type eval(
    scalar_dmat_mult_expr<ScalExpr, DMatExpr> const& expr,
    bool eval_trans = true);

template <typename LhsExpr, typename RhsExpr>
struct dvec_dvec_add_expr;

template <typename LhsExpr, typename RhsExpr>
inline decltype(auto) eval(dvec_dvec_add_expr<LhsExpr, RhsExpr> const& expr);

template <typename LhsExpr, typename RhsExpr>
inline decltype(auto) eval(dvec_dvec_add_expr<LhsExpr, RhsExpr> const& expr);

template <typename Expr,
          typename VectorType,
          VectorType (*Function)(typename VectorType::parent_type const& vec)>
struct dvec_elemwise_expr;

template <typename Expr,
          typename VectorType,
          VectorType (*Function)(typename VectorType::parent_type const& vec)>
inline decltype(auto) eval(
    dvec_elemwise_expr<Expr, VectorType, Function> const& expr);

template <typename Type>
struct scalar_expr;

template <typename Scalar>
inline typename std::enable_if<is_scalar<Scalar>::value, Scalar>::type eval(
    Scalar scalar);
}  // namespace mgcpp
#endif
