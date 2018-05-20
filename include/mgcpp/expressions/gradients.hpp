#ifndef GRADIENTS_HPP
#define GRADIENTS_HPP

#include <mgcpp/expressions/constant_expr.hpp>
#include <mgcpp/expressions/dmat_dmat_add.hpp>
#include <mgcpp/expressions/generic_expr.hpp>
#include <mgcpp/expressions/placeholder.hpp>

namespace mgcpp {

template <typename Expr,
          size_t PlaceholderID,
          template <typename> class PhResultExprType,
          typename PhResultType>
inline auto grad(
    scalar_expr<Expr> const& expr,
    placeholder_node<PlaceholderID, PhResultExprType, PhResultType> wrt);

}  // namespace mgcpp

#include <mgcpp/expressions/gradients.tpp>
#endif  // GRADIENTS_HPP
