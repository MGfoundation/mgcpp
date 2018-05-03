
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef GENERIC_OP_TPP
#define GENERIC_OP_TPP

#include <mgcpp/expressions/evaluator.hpp>
#include <mgcpp/expressions/generic_op.hpp>

namespace mgcpp {

template <int OpID,
          template <typename> class ResultExprType,
          typename ResultType,
          typename... OperandTypes>
inline decltype(auto)
generic_op<OpID, ResultExprType, ResultType, OperandTypes...>::first() const
    noexcept {
  return std::get<0>(exprs);
}

template <int OpID,
          template <typename> class ResultExprType,
          typename ResultType,
          typename... OperandTypes>
inline decltype(auto)
generic_op<OpID, ResultExprType, ResultType, OperandTypes...>::second() const
    noexcept {
  return std::get<1>(exprs);
}

template <int OpID,
          template <typename> class ResultExprType,
          typename ResultType,
          typename... OperandTypes>
typename generic_op<OpID, ResultExprType, ResultType, OperandTypes...>::
    result_type
    generic_op<OpID, ResultExprType, ResultType, OperandTypes...>::eval(
        eval_context& ctx) const {
  ctx.total_computations++;
  if (cache_ptr.use_count() > 1) {
    if (*cache_ptr == nullptr) {
      *cache_ptr = std::make_unique<result_type>(evaluator::eval(*this, ctx));
    } else {
      ctx.cache_hits++;
    }
    return **cache_ptr;
  } else {
    return evaluator::eval(*this, ctx);
  }
}

}  // namespace mgcpp

#endif  // GENERIC_OP_TPP
