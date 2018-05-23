#ifndef DMAT_REDUCE_HPP
#define DMAT_REDUCE_HPP

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/expressions/scalar_expr.hpp>

namespace mgcpp {

template <typename Expr>
struct dmat_reduce_sum_expr
    : generic_expr<dmat_reduce_sum_expr<Expr>,
                   scalar_expr,
                   typename Expr::result_type::value_type,
                   0,
                   Expr> {
  using generic_expr<dmat_reduce_sum_expr<Expr>,
                     scalar_expr,
                     typename Expr::result_type::value_type,
                     0,
                     Expr>::generic_expr;

  template <typename GradsType>
  auto grad(scalar_expr<GradsType> const& grads) const;
};

template <typename Expr>
inline decltype(auto) reduce_sum(dmat_expr<Expr> const& expr) noexcept;

template <typename Expr>
struct dmat_reduce_mean_expr
    : generic_expr<dmat_reduce_mean_expr<Expr>,
                   scalar_expr,
                   typename Expr::result_type::value_type,
                   0,
                   Expr> {
  using generic_expr<dmat_reduce_mean_expr<Expr>,
                     scalar_expr,
                     typename Expr::result_type::value_type,
                     0,
                     Expr>::generic_expr;
};

template <typename Expr>
inline decltype(auto) reduce_mean(dmat_expr<Expr> const& expr) noexcept;
}  // namespace mgcpp

#include <mgcpp/expressions/dmat_reduce_expr.tpp>
#endif  // DMAT_REDUCE_HPP
