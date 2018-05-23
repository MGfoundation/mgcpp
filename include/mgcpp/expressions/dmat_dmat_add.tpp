

//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dmat_dmat_add.hpp>

#include <mgcpp/expressions/constant_expr.hpp>
#include <mgcpp/expressions/scalar_dmat_mult.hpp>

namespace mgcpp {

template <typename LhsExpr, typename RhsExpr>
template <typename GradsType>
inline auto dmat_dmat_add_expr<LhsExpr, RhsExpr>::grad(
    dmat_expr<GradsType> const& grads) const {
  // returns (dmat, dmat)
  return std::make_tuple(~grads, ~grads);
}

namespace internal {
template <typename LhsExpr, typename RhsExpr>
auto dmat_dmat_add_impl(dmat_expr<LhsExpr> const& lhs,
                        dmat_expr<RhsExpr> const& rhs) noexcept {
  return dmat_dmat_add_expr<LhsExpr, RhsExpr>(~lhs, ~rhs);
}

// TODO: check shape compatibility??
template <typename LhsExpr, typename RhsExpr>
auto dmat_dmat_add_impl(zeros_mat_expr<LhsExpr> const&,
                        dmat_expr<RhsExpr> const& rhs) noexcept {
  return ~rhs;
}

template <typename LhsExpr, typename RhsExpr>
auto dmat_dmat_add_impl(dmat_expr<LhsExpr> const& lhs,
                        zeros_mat_expr<RhsExpr> const&) noexcept {
  return ~lhs;
}
}  // namespace internal

template <typename LhsExpr, typename RhsExpr>
auto operator+(dmat_expr<LhsExpr> const& lhs,
               dmat_expr<RhsExpr> const& rhs) noexcept {
  return internal::dmat_dmat_add_impl(~lhs, ~rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto add(dmat_expr<LhsExpr> const& lhs,
         dmat_expr<RhsExpr> const& rhs) noexcept {
  return internal::dmat_dmat_add_impl(~lhs, ~rhs);
}

template <typename LhsExpr, typename RhsExpr>
inline auto operator-(dmat_expr<LhsExpr> const& lhs,
                      dmat_expr<RhsExpr> const& rhs) noexcept {
  return lhs + static_cast<typename LhsExpr::result_type::value_type>(-1) * rhs;
}

template <typename LhsExpr, typename RhsExpr>
inline auto sub(dmat_expr<LhsExpr> const& lhs,
                dmat_expr<RhsExpr> const& rhs) noexcept {
  return lhs + static_cast<typename LhsExpr::result_type::value_type>(-1) * rhs;
}
}  // namespace mgcpp
