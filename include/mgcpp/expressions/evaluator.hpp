
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include <mgcpp/expressions/forward.hpp>

namespace mgcpp {
struct evaluator {
  template <typename Op>
  inline static typename Op::result_type eval(Op const& op, eval_context const& ctx);
};
}  // namespace mgcpp

#include <mgcpp/expressions/evaluator.tpp>
#endif  // EVALUATOR_HPP
