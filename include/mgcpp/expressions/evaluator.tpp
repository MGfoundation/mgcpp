
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/evaluator.hpp>

#include <mgcpp/expressions/dmat_dmat_add.hpp>
#include <mgcpp/expressions/dmat_dmat_mult.hpp>
#include <mgcpp/expressions/dmat_dvec_mult.hpp>
#include <mgcpp/expressions/dmat_ref_expr.hpp>
#include <mgcpp/expressions/dmat_trans_expr.hpp>
#include <mgcpp/expressions/dvec_dvec_add.hpp>
#include <mgcpp/expressions/dvec_map.hpp>
#include <mgcpp/expressions/dvec_reduce_expr.hpp>
#include <mgcpp/expressions/dvec_ref_expr.hpp>
#include <mgcpp/expressions/scalar_dmat_mult.hpp>

#include <mgcpp/operations/add.hpp>
#include <mgcpp/operations/gemm.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/operations/trans.hpp>

namespace mgcpp {
namespace internal {

template <typename LhsExpr, typename RhsExpr>
auto eval(mat_mat_add_op<LhsExpr, RhsExpr> const& expr, eval_context& ctx) {
  auto const& lhs = mgcpp::eval(expr.first(), ctx);
  auto const& rhs = mgcpp::eval(expr.second(), ctx);

  return strict::add(lhs, rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto eval(mat_mat_mult_op<LhsExpr, RhsExpr> const& expr, eval_context& ctx) {
  auto const& lhs = mgcpp::eval(expr.first(), ctx);
  auto const& rhs = mgcpp::eval(expr.second(), ctx);

  return strict::mult(lhs, rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto eval(mat_vec_mult_op<LhsExpr, RhsExpr> const& expr, eval_context& ctx) {
  auto const& lhs = mgcpp::eval(expr.first(), ctx);
  auto const& rhs = mgcpp::eval(expr.second(), ctx);

  return strict::mult(lhs, rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto eval(vec_vec_add_op<LhsExpr, RhsExpr> const& expr, eval_context& ctx) {
  auto const& lhs = mgcpp::eval(expr.first(), ctx);
  auto const& rhs = mgcpp::eval(expr.second(), ctx);

  return strict::add(lhs, rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto eval(scalar_dmat_mult_expr<LhsExpr, RhsExpr> const& expr,
          eval_context& ctx) {
  auto const& lhs = mgcpp::eval(expr.first(), ctx);
  auto const& rhs = mgcpp::eval(expr.second(), ctx);

  return strict::mult(lhs, rhs);
}

template <typename Expr>
auto eval(dmat_trans_expr<Expr> const& expr, eval_context& ctx) {
  return mgcpp::strict::trans(mgcpp::eval(expr.first(), ctx));
}

template <typename Expr,
          typename Expr::result_type (*Function)(
              typename Expr::result_type::parent_type const& vec)>
auto eval(dvec_map_expr<Expr, Function> const& expr, eval_context& ctx) {
  return Function(mgcpp::eval(expr.first(), ctx));
}

template <typename Expr,
          typename Expr::result_type::value_type (*Function)(
              typename Expr::result_type::parent_type const& vec)>
auto eval(dvec_reduce_expr<Expr, Function> const& expr, eval_context& ctx) {
  return Function(mgcpp::eval(expr.first(), ctx));
}

template <typename Matrix>
auto eval(dmat_ref_expr<Matrix> const& expr, eval_context&) {
  return expr.first();
}

template <typename Vector>
auto eval(dvec_ref_expr<Vector> const& expr, eval_context&) {
  return expr.first();
}

template <int PlaceholderID,
          template <typename> class ResultExprType,
          typename ResultType>
ResultType eval(placeholder_node<PlaceholderID, ResultExprType, ResultType>, eval_context& ctx) {
  return ctx.get_placeholder<PlaceholderID, ResultType>();
}

}  // namespace internal

template <typename Op>
auto evaluator::eval(Op const& op, eval_context& ctx) {
  return internal::eval(op, ctx);
}

}  // namespace mgcpp
