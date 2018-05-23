
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dvec_reduce_expr.hpp>
#include <mgcpp/operations/mean.hpp>
#include <mgcpp/operations/sum.hpp>

namespace mgcpp {

template <typename Expr>
template <typename GradsType>
inline auto dvec_reduce_sum_expr<Expr>::grad(
    scalar_expr<GradsType> const& grads) const {
  // returns (dvec)
  return std::make_tuple((~grads) * mgcpp::make_ones_like(this->first()));
}

template <typename Expr>
decltype(auto) reduce_sum(const dvec_expr<Expr>& expr) noexcept {
  return dvec_reduce_sum_expr<Expr>(~expr);
}

template <typename Expr>
decltype(auto) reduce_mean(const dvec_expr<Expr>& expr) noexcept {
  return dvec_reduce_mean_expr<Expr>(~expr);
}
}  // namespace mgcpp
