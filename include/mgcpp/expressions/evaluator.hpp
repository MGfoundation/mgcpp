
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include <mgcpp/expressions/forward.hpp>

namespace mgcpp {
namespace evaluator {
  template <typename T>
  inline typename T::result_type eval(expression<T> const& op, eval_context const& ctx);
};

template <typename T>
inline typename T::result_type eval(expression<T> const& expr,
                                    eval_context const& ctx);

template <typename T>
inline typename T::result_type eval(expression<T> const& expr);

}  // namespace mgcpp

#include <mgcpp/expressions/evaluator.tpp>
#endif  // EVALUATOR_HPP
