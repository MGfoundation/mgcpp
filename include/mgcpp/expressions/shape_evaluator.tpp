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
  mgcpp::shape<2> sh0 = expr.first().shape();
  mgcpp::shape<2> sh1 = expr.first().shape();
  if (sh0 != sh1)
    MGCPP_THROW_LENGTH_ERROR("incompatible shapes");
  return sh0;
}

template <typename LhsExpr, typename RhsExpr>
auto shape(dmat_dmat_mult_expr<LhsExpr, RhsExpr> const& expr,
           eval_context const& ctx) {
  mgcpp::shape<2> sh0 = expr.first().shape();
  mgcpp::shape<2> sh1 = expr.first().shape();
  if (sh0[1] != sh1[0])
    MGCPP_THROW_LENGTH_ERROR("incompatible shapes");
  return mgcpp::make_shape(sh0[0], sh1[1]);
}

template <typename LhsExpr, typename RhsExpr>
auto shape(dmat_dvec_mult_expr<LhsExpr, RhsExpr> const& expr,
           eval_context const& ctx) {
  mgcpp::shape<2> mat_shape = expr.first().shape();
  mgcpp::shape<1> vec_shape = expr.second().shape();
  if (mat_shape[1] == vec_shape[0])
    MGCPP_THROW_LENGTH_ERROR("incompatible shapes");
  return mgcpp::make_shape(mat_shape[0]);
}

template <typename LhsExpr, typename RhsExpr>
auto shape(dvec_dvec_add_expr<LhsExpr, RhsExpr> const& expr,
           eval_context const& ctx) {
  return expr.first().shape();
}

template <typename LhsExpr, typename RhsExpr>
auto shape(scalar_dmat_mult_expr<LhsExpr, RhsExpr> const& expr,
           eval_context const& ctx) {
  return expr.first().shape();
}

template <typename Expr>
auto shape(dmat_trans_expr<Expr> const& expr, eval_context const& ctx) {
  auto sh = expr.first().shape();
  std::swap(sh[0], sh[1]);
  return sh;
}

template <typename Expr>
auto shape(dvec_map_expr<Expr> const& expr, eval_context const& ctx) {
  return expr.first().shape();
}

template <typename Matrix>
auto shape(dmat_ref_expr<Matrix> const& expr, eval_context const&) {
  return expr.first().shape();
}

template <typename Vector>
auto shape(dvec_ref_expr<Vector> const& expr, eval_context const&) {
  // FIXME: first make device_vector::shape() return an mgcpp::shape<1>
  return mgcpp::make_shape(expr.first().shape());
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
