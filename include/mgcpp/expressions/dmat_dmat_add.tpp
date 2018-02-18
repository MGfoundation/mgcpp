

//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>

#include <mgcpp/expressions/dmat_dmat_add.hpp>
#include <mgcpp/expressions/dmat_dmat_mult.hpp>
#include <mgcpp/expressions/scalar_dmat_mult.hpp>
#include <mgcpp/expressions/dmat_trans_expr.hpp>
#include <mgcpp/operations/add.hpp>
#include <mgcpp/operations/gemm.hpp>
#include <mgcpp/system/assert.hpp>

#include <utility>

namespace mgcpp {
namespace internal {
template <typename LhsExpr, typename RhsExpr>
inline decltype(auto) dmat_dmat_add_subgraph_matcher(
    dmat_dmat_add_expr<LhsExpr, RhsExpr> const& expr) {
  auto lhs = mgcpp::eval(expr._lhs);
  auto rhs = mgcpp::eval(expr._rhs);

  return strict::add(lhs, rhs);
}

template <typename Expr>
decltype(auto) get_trans_mode(Expr const& expr) {
  return std::make_pair(strict::trans_mode::same, expr);
}

template <typename Expr>
decltype(auto) get_trans_mode(dmat_trans_expr<Expr> const& expr) {
  return std::make_pair(strict::trans_mode::transposed, expr._mat);
}

template <typename Expr>
decltype(auto) get_mul_constant(Expr const& expr) {
  return std::make_pair(1.0f, expr);
}

template <typename Scalar, typename Expr>
decltype(auto) get_mul_constant(
    scalar_dmat_mult_expr<Scalar, Expr> const& expr) {
  return std::make_pair(expr._scal_expr, expr._dmat_expr);
}

template <typename AType, typename BType, typename CType>
inline decltype(auto) dmat_dmat_add_subgraph_matcher(
    dmat_dmat_add_expr<dmat_dmat_mult_expr<AType, BType>, CType> const& expr) {
  auto const& tA = get_trans_mode(expr._lhs._lhs);
  auto A_trans_mode = tA.first;
  auto const& A = mgcpp::eval(tA.second);

  auto const& tB = get_trans_mode(expr._lhs._rhs);
  auto B_trans_mode = tB.first;
  auto const& B = mgcpp::eval(tB.second);

  auto const& C = mgcpp::eval(expr._rhs);
  return strict::gemm(1.0f, A_trans_mode, B_trans_mode, A, B, 1.0f, C);
}

template <typename AType, typename BType, typename CType>
inline decltype(auto) dmat_dmat_add_subgraph_matcher(
    dmat_dmat_add_expr<CType, dmat_dmat_mult_expr<AType, BType>> const& expr) {
    auto const& tA = get_trans_mode(expr._rhs._lhs);
    auto A_trans_mode = tA.first;
    auto const& A = mgcpp::eval(tA.second);

    auto const& tB = get_trans_mode(expr._rhs._rhs);
    auto B_trans_mode = tB.first;
    auto const& B = mgcpp::eval(tB.second);

    auto const& C = mgcpp::eval(expr._lhs);
    return strict::gemm(1.0f, A_trans_mode, B_trans_mode, A, B, 1.0f, C);
}

template <typename AlphaType, typename AType, typename BType, typename CType>
inline decltype(auto) dmat_dmat_add_subgraph_matcher(
    dmat_dmat_add_expr<
        scalar_dmat_mult_expr<AlphaType, dmat_dmat_mult_expr<AType, BType>>,
        CType> const& expr) {
  auto const& alpha = mgcpp::eval(expr._lhs._scal_expr);

  auto const& tA = get_trans_mode(expr._lhs._dmat_expr._lhs);
  auto A_trans_mode = tA.first;
  auto const& A = mgcpp::eval(tA.second);

  auto const& tB = get_trans_mode(expr._lhs._dmat_expr._rhs);
  auto B_trans_mode = tB.first;
  auto const& B = mgcpp::eval(tB.second);

  auto const& tC = get_mul_constant(expr._rhs._dmat_expr);
  auto beta = tC.first;
  auto const& C = tC.second;

  return strict::gemm(alpha, A_trans_mode, B_trans_mode, A, B, beta, C);
}

template <typename AlphaType, typename AType, typename BType, typename CType>
inline decltype(auto) dmat_dmat_add_subgraph_matcher(
    dmat_dmat_add_expr<
        CType,
        scalar_dmat_mult_expr<AlphaType,
                              dmat_dmat_mult_expr<AType, BType>>> const& expr) {
  auto const& alpha = mgcpp::eval(expr._rhs._scal_expr);

  auto const& tA = get_trans_mode(expr._rhs._dmat_expr._lhs);
  auto A_trans_mode = tA.first;
  auto const& A = mgcpp::eval(tA.second);

  auto const& tB = get_trans_mode(expr._rhs._dmat_expr._rhs);
  auto B_trans_mode = tB.first;
  auto const& B = mgcpp::eval(tB.second);

  auto const& tC = get_mul_constant(expr._lhs._dmat_expr);
  auto beta = tC.first;
  auto const& C = tC.second;

  return strict::gemm(alpha, A_trans_mode, B_trans_mode, A, B, beta, C);
}
}  // namespace internal

template <typename LhsExpr, typename RhsExpr>
inline dmat_dmat_add_expr<LhsExpr, RhsExpr>::dmat_dmat_add_expr(
    LhsExpr const& lhs,
    RhsExpr const& rhs) noexcept
    : _lhs(lhs), _rhs(rhs) {}

template <typename LhsExpr, typename RhsExpr>
inline dmat_dmat_add_expr<LhsExpr, RhsExpr>::dmat_dmat_add_expr(
    LhsExpr&& lhs,
    RhsExpr&& rhs) noexcept
    : _lhs(std::move(lhs)), _rhs(std::move(rhs)) {}

template <typename LhsExpr, typename RhsExpr>
decltype(auto) dmat_dmat_add_expr<LhsExpr, RhsExpr>::eval() const {
  return internal::dmat_dmat_add_subgraph_matcher(*this);
}

template <typename LhsExpr, typename RhsExpr>
typename dmat_dmat_add_expr<LhsExpr, RhsExpr>::result_type eval(
    dmat_dmat_add_expr<LhsExpr, RhsExpr> const& expr) {
  return expr.eval();
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
