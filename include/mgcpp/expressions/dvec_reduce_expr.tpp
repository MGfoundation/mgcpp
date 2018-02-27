#include <mgcpp/expressions/dvec_reduce_expr.hpp>
#include <mgcpp/operations/sum.hpp>
#include <mgcpp/operations/mean.hpp>

namespace mgcpp
{
template <
    typename Expr,
    typename Expr::result_type::value_type (*Function)(
        typename Expr::result_type::parent_type const& vec)>
dvec_reduce_expr<Expr, Function>::dvec_reduce_expr(const Expr& expr) noexcept
    : _expr(expr)
{}

template <
    typename Expr,
    typename Expr::result_type::value_type (*Function)(
        typename Expr::result_type::parent_type const& vec)>
dvec_reduce_expr<Expr, Function>::dvec_reduce_expr(Expr&& expr) noexcept
    : _expr(std::move(expr))
{}

template <
    typename Expr,
    typename Expr::result_type::value_type (*Function)(
        typename Expr::result_type::parent_type const& vec)>
decltype(auto) dvec_reduce_expr<Expr, Function>::eval() const
{
    return Function(mgcpp::eval(_expr));
}

template <typename Expr>
decltype(auto) reduce_sum(const dvec_expr<Expr>& expr) noexcept
{
    return dvec_reduce_expr<Expr, strict::sum>(~expr);
}

template <typename Expr>
decltype(auto) reduce_mean(const dvec_expr<Expr>& expr) noexcept
{
    return dvec_reduce_expr<Expr, strict::mean>(~expr);
}
}
