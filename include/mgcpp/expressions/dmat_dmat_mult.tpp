
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>

#include <mgcpp/expressions/dmat_dmat_mult.hpp>
#include <mgcpp/expressions/forward.hpp>
#include <mgcpp/global/shape.hpp>
#include <mgcpp/operations/gemm.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/system/assert.hpp>

namespace mgcpp {
namespace internal {
template <typename LhsExpr, typename RhsExpr>
inline decltype(auto) dmat_dmat_mult_subgraph_matcher(
    dmat_dmat_mult_expr<LhsExpr, RhsExpr> const& expr) {
  auto lhs = mgcpp::eval(expr._lhs);
  auto rhs = mgcpp::eval(expr._rhs);

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

  auto alpha = mgcpp::eval(expr._lhs._scal_expr);
  auto A = mgcpp::eval(expr._lhs._dmat_expr);
  auto B = mgcpp::eval(expr._rhs);

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

  auto alpha = mgcpp::eval(expr._rhs._scal_expr);
  auto A = mgcpp::eval(expr._lhs);
  auto B = mgcpp::eval(expr._rhs._dmat_expr);

  size_t m = A.shape()[0];
  size_t n = B.shape()[1];

  return strict::gemm(alpha, A, B, value_type(),
                      result_type({m, n}, value_type()));
}
}  // namespace internal

template <typename LhsExpr, typename RhsExpr>
dmat_dmat_mult_expr<LhsExpr, RhsExpr>::dmat_dmat_mult_expr(
    LhsExpr const& lhs,
    RhsExpr const& rhs) noexcept
    : _lhs(lhs), _rhs(rhs) {}

template <typename LhsExpr, typename RhsExpr>
typename dmat_dmat_mult_expr<LhsExpr, RhsExpr>::result_type
dmat_dmat_mult_expr<LhsExpr, RhsExpr>::eval() const {
  return internal::dmat_dmat_mult_subgraph_matcher(*this);
}

template <typename LhsExpr, typename RhsExpr>
decltype(auto) eval(
    dmat_dmat_mult_expr<LhsExpr, RhsExpr> const& expr) {
  return expr.eval();
}

template <typename LhsExpr, typename RhsExpr>
dmat_dmat_mult_expr<LhsExpr, RhsExpr> operator*(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept {
  return dmat_dmat_mult_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
}

template <typename LhsExpr, typename RhsExpr>
dmat_dmat_mult_expr<LhsExpr, RhsExpr> mult(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept {
  return dmat_dmat_mult_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
}
}  // namespace mgcpp
