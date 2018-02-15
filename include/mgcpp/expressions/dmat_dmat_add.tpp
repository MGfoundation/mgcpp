

//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>

#include <mgcpp/expressions/dmat_dmat_add.hpp>
#include <mgcpp/expressions/dmat_dmat_mult.hpp>
#include <mgcpp/expressions/forward.hpp>
#include <mgcpp/operations/add.hpp>
#include <mgcpp/operations/gemm.hpp>
#include <mgcpp/system/assert.hpp>

namespace mgcpp {
namespace internal {
template <typename LhsExpr, typename RhsExpr>
inline decltype(auto) dmat_dmat_add_subgraph_matcher(
    dmat_dmat_add_expr<LhsExpr, RhsExpr> const& expr) {
  auto const& lhs = mgcpp::eval(expr._lhs, false);
  auto const& rhs = mgcpp::eval(expr._rhs, false);

  return strict::add(lhs, rhs);
}

template <typename AType, typename BType, typename CType>
inline decltype(auto) dmat_dmat_subgraph_matcher(
    dmat_dmat_add_expr<dmat_dmat_mult_expr<AType, BType>, CType> const& expr) {
  auto const& A = mgcpp::eval(expr._lhs._lhs, false);
  auto const& B = mgcpp::eval(expr._lhs._rhs, false);
  auto const& C = mgcpp::eval(expr._rhs);
  return strict::gemm(A, B, C);
}

template <typename AType, typename BType, typename CType>
inline decltype(auto) dmat_dmat_subgraph_matcher(
    dmat_dmat_add_expr<CType, dmat_dmat_mult_expr<AType, BType>> const& expr) {
  auto const& A = mgcpp::eval(expr._rhs._lhs, false);
  auto const& B = mgcpp::eval(expr._rhs._rhs, false);
  auto const& C = mgcpp::eval(expr._lhs);
  return strict::gemm(A, B, C);
}

template <typename AlphaType, typename AType, typename BType, typename CType>
inline decltype(auto) dmat_dmat_subgraph_matcher(
    dmat_dmat_add_expr<
        scalar_dmat_mult_expr<AlphaType, dmat_dmat_mult_expr<AType, BType>>,
        CType> const& expr) {
  auto const& alpha = mgcpp::eval(expr._rhs._scal_expr);
  auto const& A = mgcpp::eval(expr._rhs._dmat_expr._lhs, false);
  auto const& B = mgcpp::eval(expr._rhs._dmat_expr._rhs, false);
  float beta = 1.0;
  auto const& C = mgcpp::eval(expr._lhs._dmat_expr);
  return strict::gemm(alpha, A, B, beta, C);
}

template <typename AlphaType, typename AType, typename BType, typename CType>
inline decltype(auto) dmat_dmat_subgraph_matcher(
    dmat_dmat_add_expr<
        CType,
        scalar_dmat_mult_expr<AlphaType,
                              dmat_dmat_mult_expr<AType, BType>>> const& expr) {
  auto const& alpha = mgcpp::eval(expr._lhs._scal_expr);
  auto const& A = mgcpp::eval(expr._lhs._dmat_expr._lhs, false);
  auto const& B = mgcpp::eval(expr._lhs._dmat_expr._rhs, false);
  float beta = 1.0;
  auto const& C = mgcpp::eval(expr._rhs._dmat_expr);
  return strict::gemm(alpha, A, B, beta, C);
}

template <typename AlphaType,
          typename AType,
          typename BType,
          typename BetaType,
          typename CType>
inline decltype(auto) dmat_dmat_subgraph_matcher(
    dmat_dmat_add_expr<
        scalar_dmat_mult_expr<AlphaType, dmat_dmat_mult_expr<AType, BType>>,
        scalar_dmat_mult_expr<BetaType, CType>> const& expr) {
  auto const& alpha = mgcpp::eval(expr._rhs._scal_expr);
  auto const& A = mgcpp::eval(expr._rhs._dmat_expr._lhs, false);
  auto const& B = mgcpp::eval(expr._rhs._dmat_expr._rhs, false);
  auto beta = mgcpp::eval(expr._lhs._scal_expr);
  auto const& C = mgcpp::eval(expr._lhs._dmat_expr);
  return strict::gemm(alpha, A, B, beta, C);
}

template <typename AlphaType,
          typename AType,
          typename BType,
          typename BetaType,
          typename CType>
inline decltype(auto) dmat_dmat_subgraph_matcher(
    dmat_dmat_add_expr<
        scalar_dmat_mult_expr<BetaType, CType>,
        scalar_dmat_mult_expr<AlphaType,
                              dmat_dmat_mult_expr<AType, BType>>> const& expr) {
  auto const& alpha = mgcpp::eval(expr._lhs._scal_expr);
  auto const& A = mgcpp::eval(expr._lhs._dmat_expr._lhs, false);
  auto const& B = mgcpp::eval(expr._lhs._dmat_expr._rhs, false);
  auto beta = mgcpp::eval(expr._rhs._scal_expr);
  auto const& C = mgcpp::eval(expr._rhs._dmat_expr, false);
  return strict::gemm(alpha, A, B, beta, C);
}
}  // namespace internal

template <typename LhsExpr, typename RhsExpr>
dmat_dmat_add_expr<LhsExpr, RhsExpr>::dmat_dmat_add_expr(
    LhsExpr const& lhs,
    RhsExpr const& rhs) noexcept
    : _lhs(lhs), _rhs(rhs) {}

template <typename LhsExpr, typename RhsExpr>
typename dmat_dmat_add_expr<LhsExpr, RhsExpr>::result_type
dmat_dmat_add_expr<LhsExpr, RhsExpr>::eval(bool eval_trans) const {
  (void)eval_trans;
  return internal::dmat_dmat_add_subgraph_matcher(*this);
}

template <typename LhsExpr, typename RhsExpr>
typename dmat_dmat_add_expr<LhsExpr, RhsExpr>::result_type eval(
    dmat_dmat_add_expr<LhsExpr, RhsExpr> const& expr,
    bool eval_trans) {
  expr.eval(eval_trans);
}

template <typename LhsExpr, typename RhsExpr>
dmat_dmat_add_expr<LhsExpr, RhsExpr> operator+(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept {
  return dmat_dmat_add_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
}

template <typename LhsExpr, typename RhsExpr>
dmat_dmat_add_expr<LhsExpr, RhsExpr> add(
    dmat_expr<LhsExpr> const& lhs,
    dmat_expr<RhsExpr> const& rhs) noexcept {
  return dmat_dmat_add_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
}
}  // namespace mgcpp
