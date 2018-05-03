
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
  apply_void(take_rest<NParameters>(exprs), [&](auto const& expr) { expr.traverse(ctx); });
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
  /*
    if (ctx.cnt.at(this->id)-- > 1)
    {
      auto it = ctx.cache.find(this->id);
      if (it != ctx.cache.end()) {
        ctx.cache_hits++;
        return it->second.template get<result_type>();
      }
      else {
        auto result = evaluator::eval(*this, ctx);
        ctx.cache[this->id] = result;
        return result;
      }
    }
  */
  return evaluator::eval(*this, ctx);
}

}  // namespace mgcpp
