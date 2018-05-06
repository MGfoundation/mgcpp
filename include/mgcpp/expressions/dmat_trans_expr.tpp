
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dmat_trans_expr.hpp>

namespace mgcpp {

template <typename Expr>
inline dmat_trans_expr<Expr> trans(dmat_expr<Expr> const& expr) noexcept {
  return dmat_trans_expr<Expr>(~expr);
}

}  // namespace mgcpp
