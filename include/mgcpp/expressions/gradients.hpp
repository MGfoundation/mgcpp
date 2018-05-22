#ifndef GRADIENTS_HPP
#define GRADIENTS_HPP

#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/expressions/placeholder.hpp>

namespace mgcpp {

template <typename Expr, size_t PlaceholderID, typename PhResultType>
inline auto grad(scalar_expr<Expr> const& expr,
                 placeholder_node<PlaceholderID, PhResultType> wrt);

}  // namespace mgcpp

#include <mgcpp/expressions/gradients.tpp>
#endif  // GRADIENTS_HPP
