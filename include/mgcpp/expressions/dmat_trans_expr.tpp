#include <mgcpp/expressions/dmat_trans_expr.hpp>
#include <mgcpp/operations/trans.hpp>

namespace mgcpp {

template <typename Expr>
dmat_trans_expr<Expr>::dmat_trans_expr(Expr const& mat) noexcept : _mat(mat) {}

template <typename Expr>
dmat_trans_expr<Expr>::dmat_trans_expr(Expr&& mat) noexcept : _mat(std::move(mat)) {}

template <typename Expr>
decltype(auto) dmat_trans_expr<Expr>::eval() const
{
    return mgcpp::strict::trans(mgcpp::eval(_mat));
}

template <typename Expr>
inline decltype(auto) eval(dmat_trans_expr<Expr> const& expr)
{
    return expr.eval();
}

template <typename Expr>
inline dmat_trans_expr<Expr> trans(dmat_expr<Expr> const& expr) noexcept
{
    return dmat_trans_expr<Expr>(~expr);
}

}
