
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_FORWARD_HPP_
#define _MGCPP_EXPRESSIONS_FORWARD_HPP_

#include <cstddef>

namespace mgcpp {

template <typename Type>
class expression;

struct eval_context;

template <typename TagType,
          size_t Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
struct generic_expr;
}

#endif
