
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_SCALAR_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_SCALAR_EXPR_HPP_

#include <mgcpp/type_traits/type_traits.hpp>
#include <type_traits>

namespace mgcpp {
template <typename Type>
struct scalar_expr;

template <typename Scalar>
inline typename std::enable_if<is_scalar<Scalar>::value, Scalar>::type eval(
    Scalar scalar) {
  return Scalar(scalar);
}
}  // namespace mgcpp

#endif
