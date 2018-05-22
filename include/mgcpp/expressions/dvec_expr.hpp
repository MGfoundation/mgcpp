
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_DENSE_VEC_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DENSE_VEC_EXPR_HPP_

#include <mgcpp/expressions/expression.hpp>
#include <mgcpp/system/concept.hpp>
#include <mgcpp/vector/forward.hpp>

#include <cstdlib>
#include <string>
#include <utility>

namespace mgcpp {
template <typename Expr>
struct dvec_expr : public expression<Expr> {};
}  // namespace mgcpp

#endif
