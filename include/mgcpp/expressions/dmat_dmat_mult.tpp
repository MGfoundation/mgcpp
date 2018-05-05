
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dmat_dmat_mult.hpp>

namespace mgcpp {
/*
namespace internal {

template <typename LhsExpr, typename RhsExpr>
inline decltype(auto) dmat_dmat_mult_subgraph_matcher(
    dmat_dmat_mult_expr<LhsExpr, RhsExpr> const& expr) {
  auto const& lhs = mgcpp::eval(expr._lhs);
  auto const& rhs = mgcpp::eval(expr._rhs);

  return strict::mult(lhs, rhs);
}

template <typename LhsScal, typename LhsMat, typename RhsExpr>
inline decltype(auto) dmat_dmat_mult_subgraph_matcher(
    dmat_dmat_mult_expr<scalar_dmat_mult_expr<LhsScal, LhsMat>, RhsExpr> const&
        expr) {
  using value_type = typename RhsExpr::value_type;
  using result_type =
      typename dmat_dmat_mult_expr<scalar_dmat_mult_expr<LhsScal, LhsMat>,
                                   RhsExpr>::result_type;

  auto const& alpha = mgcpp::eval(expr._lhs._scal_expr);
  auto const& A = mgcpp::eval(expr._lhs._dmat_expr);
  auto const& B = mgcpp::eval(expr._rhs);

  size_t m = A.shape()[0];
  size_t n = B.shape()[1];

  return strict::gemm(alpha, A, B, value_type(),
                      result_type({m, n}, value_type()));
}

template <typename RhsScal, typename RhsMat, typename LhsExpr>
inline decltype(auto) dmat_dmat_mult_subgraph_matcher(
    dmat_dmat_mult_expr<LhsExpr, scalar_dmat_mult_expr<RhsScal, RhsMat>> const&
        expr) {
  using value_type = typename LhsExpr::value_type;
  using result_type = typename dmat_dmat_mult_expr<
      LhsExpr, scalar_dmat_mult_expr<RhsScal, RhsMat>>::result_type;

  auto const& alpha = mgcpp::eval(expr._rhs._scal_expr);
  auto const& A = mgcpp::eval(expr._lhs);
  auto const& B = mgcpp::eval(expr._rhs._dmat_expr);

  size_t m = A.shape()[0];
  size_t n = B.shape()[1];

  return strict::gemm(alpha, A, B, value_type(),
                      result_type({m, n}, value_type()));
}
}  // namespace internal

template <typename LhsExpr, typename RhsExpr>
inline dmat_dmat_mult_expr<LhsExpr, RhsExpr>::dmat_dmat_mult_expr(
    LhsExpr const& lhs,
    RhsExpr const& rhs) noexcept
    : _lhs(lhs), _rhs(rhs) {}

template <typename LhsExpr, typename RhsExpr>
inline dmat_dmat_mult_expr<LhsExpr, RhsExpr>::dmat_dmat_mult_expr(
    LhsExpr&& lhs,
    RhsExpr&& rhs) noexcept
    : _lhs(std::move(lhs)), _rhs(std::move(rhs)) {}

template <typename LhsExpr, typename RhsExpr>
typename dmat_dmat_mult_expr<LhsExpr, RhsExpr>::result_type
dmat_dmat_mult_expr<LhsExpr, RhsExpr>::eval() const {
  return internal::dmat_dmat_mult_subgraph_matcher(*this);
}
*/

template <typename LhsExpr, typename RhsExpr>
mat_mat_mult_op<LhsExpr, RhsExpr> operator*(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept {
  return mat_mat_mult_op<LhsExpr, RhsExpr>(~lhs, ~rhs);
}

template <typename LhsExpr, typename RhsExpr>
mat_mat_mult_op<LhsExpr, RhsExpr> mult(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept {
  return mat_mat_mult_op<LhsExpr, RhsExpr>(~lhs, ~rhs);
}
}  // namespace mgcpp
