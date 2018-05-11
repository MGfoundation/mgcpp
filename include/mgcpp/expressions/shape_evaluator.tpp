#include <mgcpp/expressions/shape_evaluator.hpp>

#include <mgcpp/expressions/constant_expr.hpp>
#include <mgcpp/expressions/dmat_dmat_add.hpp>
#include <mgcpp/expressions/dmat_dmat_mult.hpp>
#include <mgcpp/expressions/dmat_dvec_mult.hpp>
#include <mgcpp/expressions/dmat_ref_expr.hpp>
#include <mgcpp/expressions/dmat_trans_expr.hpp>
#include <mgcpp/expressions/dvec_dvec_add.hpp>
#include <mgcpp/expressions/dvec_map.hpp>
#include <mgcpp/expressions/dvec_reduce_expr.hpp>
#include <mgcpp/expressions/dvec_ref_expr.hpp>
#include <mgcpp/expressions/placeholder.hpp>
#include <mgcpp/expressions/scalar_dmat_mult.hpp>
#include <mgcpp/expressions/tie_expr.hpp>

namespace mgcpp {

namespace internal {

template <typename LhsExpr, typename RhsExpr>
auto shape(dmat_dmat_add_expr<LhsExpr, RhsExpr> const& expr,
           eval_context const& ctx) {
  // TODO
}

template <typename LhsExpr, typename RhsExpr>
auto shape(dmat_dmat_mult_expr<LhsExpr, RhsExpr> const& expr,
           eval_context const& ctx) {
  // TODO
}

template <typename LhsExpr, typename RhsExpr>
auto shape(dmat_dvec_mult_expr<LhsExpr, RhsExpr> const& expr,
           eval_context const& ctx) {
  // TODO
}

template <typename LhsExpr, typename RhsExpr>
auto shape(dvec_dvec_add_expr<LhsExpr, RhsExpr> const& expr,
           eval_context const& ctx) {
  // TODO
}

template <typename LhsExpr, typename RhsExpr>
auto shape(scalar_dmat_mult_expr<LhsExpr, RhsExpr> const& expr,
           eval_context const& ctx) {
  // TODO
}

template <typename Expr>
auto shape(dmat_trans_expr<Expr> const& expr, eval_context const& ctx) {
  // TODO
}

template <typename Expr>
auto shape(dvec_map_expr<Expr> const& expr, eval_context const& ctx) {
  // TODO
}

template <typename Matrix>
auto shape(dmat_ref_expr<Matrix> const& expr, eval_context const&) {
  // TODO
}

template <typename Vector>
auto shape(dvec_ref_expr<Vector> const& expr, eval_context const&) {
  // TODO
}

template <int PlaceholderID,
          template <typename> class ResultExprType,
          typename ResultType>
auto shape(placeholder_node<PlaceholderID, ResultExprType, ResultType>,
                 eval_context const& ctx) {
  return ctx.get_placeholder<PlaceholderID, ResultType>().shape();
}

template <typename Expr>
auto shape(zeros_mat_expr<Expr> const& expr, eval_context const& ctx) {
  return mgcpp::eval(expr.first(), ctx);
}

template <typename Expr>
auto shape(ones_mat_expr<Expr> const& expr, eval_context const& ctx) {
  return mgcpp::eval(expr.first(), ctx);
}

}  // namespace internal

template <typename Op>
typename Op::result_type::shape_type shape_evaluator::shape(const Op& op,
                                                const eval_context& ctx) {
  return internal::shape(op, ctx);
}

}  // namespace mgcpp
