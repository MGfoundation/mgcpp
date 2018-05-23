
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dmat_dvec_mult.hpp>

#include <mgcpp/expressions/dvec_dvec_outer.hpp>

namespace mgcpp {

template <typename LhsExpr, typename RhsExpr>
template <typename GradsType>
auto dmat_dvec_mult_expr<LhsExpr, RhsExpr>::grad(
    dvec_expr<GradsType> const& grads) const {
  return std::make_tuple(mgcpp::outer(~grads, this->second()),
                         mgcpp::trans(this->first()) * (~grads));
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
