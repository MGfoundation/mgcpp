
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/generic_expr.hpp>

#include <mgcpp/expressions/eval_cache.hpp>
#include <mgcpp/expressions/evaluator.hpp>
#include <mgcpp/expressions/expression.hpp>
#include <mgcpp/expressions/scalar_expr.hpp>
#include <mgcpp/global/tuple_utils.hpp>
#include <mgcpp/system/assert.hpp>

namespace mgcpp {

template <typename TagType,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
inline generic_expr<
    TagType,
    ResultExprType,
    ResultType,
    NParameters,
    OperandTypes...>::generic_expr(OperandTypes... args) noexcept
    : exprs(std::move(args)...) {}

template <typename TagType,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
void generic_expr<TagType,
                  ResultExprType,
                  ResultType,
                  NParameters,
                  OperandTypes...>::traverse() const {
  analyze_graph(this->id, [&] {
    apply_void(this->operands(), [](auto const& ch) { ch.traverse(); });
  });
}

template <typename TagType,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
inline decltype(auto) generic_expr<TagType,
                                   ResultExprType,
                                   ResultType,
                                   NParameters,
                                   OperandTypes...>::operands() const noexcept {
  return take_rest<NParameters>(exprs);
}

template <typename TagType,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
inline decltype(auto) generic_expr<TagType,
                                   ResultExprType,
                                   ResultType,
                                   NParameters,
                                   OperandTypes...>::parameters() const
    noexcept {
  return take_front<NParameters>(exprs);
}

template <typename TagType,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
inline decltype(auto) generic_expr<TagType,
                                   ResultExprType,
                                   ResultType,
                                   NParameters,
                                   OperandTypes...>::first() const noexcept {
  return std::get<0>(exprs);
}

template <typename TagType,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
inline decltype(auto) generic_expr<TagType,
                                   ResultExprType,
                                   ResultType,
                                   NParameters,
                                   OperandTypes...>::second() const noexcept {
  return std::get<1>(exprs);
}

template <typename TagType,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
ResultType generic_expr<TagType,
                        ResultExprType,
                        ResultType,
                        NParameters,
                        OperandTypes...>::eval() const {
  eval_context ctx;
  return this->eval(ctx);
}

template <typename TagType,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
ResultType generic_expr<TagType,
                        ResultExprType,
                        ResultType,
                        NParameters,
                        OperandTypes...>::eval(eval_context const& ctx) const {
  static_any result = evaluate_if_needed(
      this->id, !is_terminal, [&] { this->traverse(); },
      [&] { return static_any(evaluator::eval(*this, ctx)); });
  return result.get<ResultType>();
}

template <typename TagType,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
template <typename T, typename>
typename T::shape_type
generic_expr<TagType,
             ResultExprType,
             ResultType,
             NParameters,
             OperandTypes...>::shape(eval_context const& ctx) const {
  return shape_evaluator::shape(*this, ctx);
}

}  // namespace mgcpp
