
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/evaluator.hpp>
#include <mgcpp/expressions/generic_op.hpp>
#include <mgcpp/global/tuple_utils.hpp>

namespace mgcpp {

template <typename TagType,
          TagType Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
inline decltype(auto)
generic_op<TagType, Tag, ResultExprType, ResultType, NParameters, OperandTypes...>::first()
    const noexcept {
  return std::get<0>(exprs);
}

template <typename TagType,
          TagType Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
inline decltype(auto)
generic_op<TagType, Tag, ResultExprType, ResultType, NParameters, OperandTypes...>::second()
    const noexcept {
  return std::get<1>(exprs);
}

template <typename TagType,
          TagType Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
inline void
generic_op<TagType, Tag, ResultExprType, ResultType, NParameters, OperandTypes...>::traverse(
    eval_context& ctx) const {
  ctx.cnt[this->id]++;

  // traverse from NParameters to sizeof...(OperandTypes) - 1
  apply_void(take_rest<NParameters>(exprs), [&](auto const& expr) { mgcpp::traverse(expr, ctx); });
}

template <typename TagType,
          TagType Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
typename generic_op<TagType, Tag, ResultExprType, ResultType, NParameters, OperandTypes...>::
    result_type
    generic_op<TagType, Tag, ResultExprType, ResultType, NParameters, OperandTypes...>::eval(
        eval_context& ctx) const {
  ctx.total_computations++;

  // try to find cache
  auto it = ctx.cache.find(this->id);

  // number of instances of this node left
  auto left = --ctx.cnt.at(this->id);

  // If cached, return the cache
  if (it != ctx.cache.end()) {
    ctx.cache_hits++;
    auto cached = std::move(it->second.template get<result_type>());

    // Erase the cache for memory if it is no longer needed
    if (left == 0)
      ctx.cache.erase(it);

    return cached;
  }

  // If the same subexpression is shared by more than 1 nodes
  // and this is not a terminal node, cache
  if (!is_terminal && left >= 1)
  {
      auto result = evaluator::eval(*this, ctx);
      ctx.cache[this->id] = result;
      return result;
  }

  // No need to cache if the expression is not shared
  return evaluator::eval(*this, ctx);
}

}  // namespace mgcpp
