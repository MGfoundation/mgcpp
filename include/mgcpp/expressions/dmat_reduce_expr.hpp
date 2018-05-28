#ifndef DMAT_REDUCE_HPP
#define DMAT_REDUCE_HPP

#include <mgcpp/expressions/dmat_expr.hpp>
#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/expressions/scalar_expr.hpp>

namespace mgcpp {

struct dmat_reduce_sum_expr_type;

template <typename Expr>
using dmat_reduce_sum_expr =
    generic_expr<dmat_reduce_sum_expr_type,
                 0,
                 scalar_expr,
                 typename Expr::result_type::value_type,
                 0,
                 Expr>;

template <typename Expr>
inline decltype(auto) reduce_sum(dmat_expr<Expr> const& expr) noexcept;

struct dmat_reduce_mean_expr_type;

template <typename Expr>
using dmat_reduce_mean_expr =
    generic_expr<dmat_reduce_mean_expr_type,
                 0,
                 scalar_expr,
                 typename Expr::result_type::value_type,
                 0,
                 Expr>;

template <typename Expr>
inline decltype(auto) reduce_mean(dmat_expr<Expr> const& expr) noexcept;
}  // namespace mgcpp

#include <mgcpp/expressions/dmat_reduce_expr.tpp>
#endif  // DMAT_REDUCE_HPP
