
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/evaluator.hpp>
#include <mgcpp/expressions/shape_evaluator.hpp>

#include <mgcpp/expressions/constant_expr.hpp>
#include <mgcpp/expressions/dmat_dmat_add.hpp>
#include <mgcpp/expressions/dmat_dmat_mult.hpp>
#include <mgcpp/expressions/dmat_dvec_mult.hpp>
#include <mgcpp/expressions/dmat_reduce_expr.hpp>
#include <mgcpp/expressions/dmat_ref_expr.hpp>
#include <mgcpp/expressions/dmat_trans_expr.hpp>
#include <mgcpp/expressions/dvec_dvec_add.hpp>
#include <mgcpp/expressions/dvec_dvec_outer.hpp>
#include <mgcpp/expressions/dvec_map.hpp>
#include <mgcpp/expressions/dvec_reduce_expr.hpp>
#include <mgcpp/expressions/dvec_ref_expr.hpp>
#include <mgcpp/expressions/placeholder.hpp>
#include <mgcpp/expressions/scalar_dmat_mult.hpp>
#include <mgcpp/expressions/scalar_dvec_mult.hpp>
#include <mgcpp/expressions/tie_expr.hpp>

#include <mgcpp/operations/add.hpp>
#include <mgcpp/operations/gemm.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/operations/outer.hpp>
#include <mgcpp/operations/trans.hpp>

namespace mgcpp {
namespace evaluator {
namespace internal {

template <typename LhsExpr, typename RhsExpr>
auto eval(dmat_dmat_add_expr<LhsExpr, RhsExpr> const& expr,
          eval_context const& ctx) {
  auto lhs = mgcpp::eval(expr.first(), ctx);
  auto rhs = mgcpp::eval(expr.second(), ctx);

  return strict::add(lhs, rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto eval(dmat_dmat_mult_expr<LhsExpr, RhsExpr> const& expr,
          eval_context const& ctx) {
  auto lhs = mgcpp::eval(expr.first(), ctx);
  auto rhs = mgcpp::eval(expr.second(), ctx);

  return strict::mult(lhs, rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto eval(dmat_dvec_mult_expr<LhsExpr, RhsExpr> const& expr,
          eval_context const& ctx) {
  auto lhs = mgcpp::eval(expr.first(), ctx);
  auto rhs = mgcpp::eval(expr.second(), ctx);

  return strict::mult(lhs, rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto eval(dvec_dvec_add_expr<LhsExpr, RhsExpr> const& expr,
          eval_context const& ctx) {
  auto lhs = mgcpp::eval(expr.first(), ctx);
  auto rhs = mgcpp::eval(expr.second(), ctx);

  return strict::add(lhs, rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto eval(dvec_dvec_outer_expr<LhsExpr, RhsExpr> const& expr,
          eval_context const& ctx) {
  auto lhs = mgcpp::eval(expr.first(), ctx);
  auto rhs = mgcpp::eval(expr.second(), ctx);

  return strict::outer(lhs, rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto eval(scalar_dmat_mult_expr<LhsExpr, RhsExpr> const& expr,
          eval_context const& ctx) {
  auto lhs = mgcpp::eval(expr.first(), ctx);
  auto rhs = mgcpp::eval(expr.second(), ctx);

  return strict::mult(lhs, rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto eval(scalar_dvec_mult_expr<LhsExpr, RhsExpr> const& expr,
          eval_context const& ctx) {
  auto lhs = mgcpp::eval(expr.first(), ctx);
  auto rhs = mgcpp::eval(expr.second(), ctx);

  return strict::mult(lhs, rhs);
}

template <typename Expr>
auto eval(dmat_trans_expr<Expr> const& expr, eval_context const& ctx) {
  return mgcpp::strict::trans(mgcpp::eval(expr.first(), ctx));
}

template <typename Expr>
auto eval(dvec_map_expr<Expr> const& expr, eval_context const& ctx) {
  return expr.first()(mgcpp::eval(expr.second(), ctx));
}

template <typename Expr>
auto eval(dvec_reduce_sum_expr<Expr> const& expr, eval_context const& ctx) {
  return strict::sum(mgcpp::eval(expr.first(), ctx));
}

template <typename Expr>
auto eval(dvec_reduce_mean_expr<Expr> const& expr, eval_context const& ctx) {
  return strict::mean(mgcpp::eval(expr.first(), ctx));
}

template <typename Expr>
auto eval(dmat_reduce_sum_expr<Expr> const& expr, eval_context const& ctx) {
  return strict::sum(mgcpp::eval(expr.first(), ctx));
}

template <typename Expr>
auto eval(dmat_reduce_mean_expr<Expr> const& expr, eval_context const& ctx) {
  return strict::mean(mgcpp::eval(expr.first(), ctx));
}

template <typename Matrix>
auto eval(dmat_ref_expr<Matrix> const& expr, eval_context const&) {
  return expr.first();
}

template <typename Vector>
auto eval(dvec_ref_expr<Vector> const& expr, eval_context const&) {
  return expr.first();
}

template <size_t PlaceholderID, typename ResultType>
ResultType eval(placeholder_node<PlaceholderID, ResultType>,
                eval_context const& ctx) {
  return ctx.get_placeholder<PlaceholderID, ResultType>();
}

template <typename... Exprs>
auto eval(tie_expr<Exprs...> const& tie, eval_context const& ctx) {
  return apply((~tie).exprs,
               [&](auto const& t) { return mgcpp::eval(t, ctx); });
}

template <typename Expr>
auto eval(zeros_mat_expr<Expr> const& expr, eval_context const& ctx) {
  auto shape = expr.first().shape(ctx);
  using result_type = typename Expr::result_type;
  using value_type = typename result_type::value_type;
  return result_type(shape, static_cast<value_type>(0.0));
}

template <typename Expr>
auto eval(ones_mat_expr<Expr> const& expr, eval_context const& ctx) {
  auto shape = expr.first().shape(ctx);
  using result_type = typename Expr::result_type;
  using value_type = typename result_type::value_type;
  return result_type(shape, static_cast<value_type>(1.0));
}

template <typename Expr>
auto eval(zeros_vec_expr<Expr> const& expr, eval_context const& ctx) {
  auto shape = expr.first().shape(ctx);
  using result_type = typename Expr::result_type;
  using value_type = typename result_type::value_type;
  return result_type(shape[0], static_cast<value_type>(0.0));
}

template <typename Expr>
auto eval(ones_vec_expr<Expr> const& expr, eval_context const& ctx) {
  auto shape = expr.first().shape(ctx);
  using result_type = typename Expr::result_type;
  using value_type = typename result_type::value_type;
  return result_type(shape[0], static_cast<value_type>(1.0));
}

template <typename T>
T eval(scalar_constant_expr<T> const& expr, eval_context const&) {
  return expr.first();
}

template <typename T>
inline auto eval(scalar_zero_constant_expr<T>, eval_context const&) {
  return static_cast<T>(0);
}

template <typename T>
inline auto eval(scalar_one_constant_expr<T>, eval_context const&) {
  return static_cast<T>(1);
}
}  // namespace internal

template <typename T>
typename T::result_type eval(expression<T> const& op, eval_context const& ctx) {
  return internal::eval(~op, ctx);
}
}  // namespace evaluator

template <typename T>
inline typename T::result_type eval(expression<T> const& expr,
                                    eval_context const& ctx) {
  return (~expr).eval(ctx);
}

template <typename T>
inline typename T::result_type eval(expression<T> const& expr) {
  return (~expr).eval();
}

}  // namespace mgcpp
