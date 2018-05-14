
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef EVAL_CONTEXT_HPP
#define EVAL_CONTEXT_HPP

#include <memory>
#include <mgcpp/global/type_erased.hpp>
#include <mgcpp/expressions/expression.hpp>
#include <mgcpp/expressions/forward.hpp>
#include <mgcpp/expressions/placeholder.hpp>
#include <type_traits>
#include <unordered_map>

namespace mgcpp {

struct eval_context {
  /** Associate a value to a placeholder. The associated value will be fed to
   * the placeholder's place when the graph is evaluated.
   * \param ph The placeholder.
   * \param val The value associated with the placeholder.
   */
  template <int Num,
            template <typename> class ResultExprType,
            typename ResultType>
  void feed(placeholder_node<Num, ResultExprType, ResultType> ph,
            ResultType const& val);

  /** Get the value of the placeholder associated with the PlaceholderID.
   * ResultType should be the type of the value when feed() was called.
   */
  template <size_t PlaceholderID, typename ResultType>
  auto get_placeholder() const;

protected:
  std::unordered_map<int, static_any> _placeholders;
};

}  // namespace mgcpp

#include <mgcpp/expressions/eval_context.tpp>
#endif  // EVAL_CONTEXT_HPP
