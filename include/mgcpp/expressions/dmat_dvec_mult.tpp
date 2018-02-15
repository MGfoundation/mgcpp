
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dmat_dvec_mult.hpp>
#include <mgcpp/operations/mult.hpp>

namespace mgcpp {
namespace internal {
template <typename MatExpr, typename VecExpr>
inline decltype(auto) dmat_dvec_mult_subgraph_matcher(
    dmat_dvec_mult_expr<MatExpr, VecExpr> const& expr) {
  auto const& mat = mgcpp::eval(expr._mat, false);
  auto const& vec = mgcpp::eval(expr._vec);

  return strict::mult(mat, vec);
}
}  // namespace internal

template <typename MatExpr, typename VecExpr>
dmat_dvec_mult_expr<MatExpr, VecExpr>::dmat_dvec_mult_expr(
    MatExpr const& mat,
    VecExpr const& vec) noexcept
    : _mat(mat), _vec(vec) {}

template <typename MatExpr, typename VecExpr>
typename dmat_dvec_mult_expr<MatExpr, VecExpr>::result_type
dmat_dvec_mult_expr<MatExpr, VecExpr>::eval() const {
  return internal::dmat_dvec_mult_subgraph_matcher(*this);
}

template <typename MatExpr, typename VecExpr>
typename dmat_dvec_mult_expr<MatExpr, VecExpr>::result_type eval(
    dmat_dvec_mult_expr<MatExpr, VecExpr> const& expr) {
  return expr.eval();
}

template <typename MatExpr, typename VecExpr>
inline dmat_dvec_mult_expr<MatExpr, VecExpr> operator*(
    dmat_expr<MatExpr> const& mat,
    dvec_expr<VecExpr> const& vec) noexcept {
  return dmat_dvec_mult_expr<MatExpr, VecExpr>(~mat, ~vec);
}

template <typename MatExpr, typename VecExpr>
inline dmat_dvec_mult_expr<MatExpr, VecExpr> mult(
    dmat_expr<MatExpr> const& mat,
    dvec_expr<VecExpr> const& vec) noexcept {
  return dmat_dvec_mult_expr<MatExpr, VecExpr>(~mat, ~vec);
}
}  // namespace mgcpp
