
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/evaluator.hpp>
#include <mgcpp/expressions/generic_op.hpp>
#include <mgcpp/global/tuple_utils.hpp>
#include <mgcpp/system/assert.hpp>

namespace mgcpp {

template <typename TagType,
          TagType Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
inline generic_op<TagType,
                  Tag,
                  ResultExprType,
                  ResultType,
                  NParameters,
                  OperandTypes...>::generic_op(OperandTypes... args) noexcept
    : exprs(std::move(args)...) {}

template <typename TagType,
          TagType Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
inline decltype(auto) generic_op<TagType,
                                 Tag,
                                 ResultExprType,
                                 ResultType,
                                 NParameters,
                                 OperandTypes...>::first() const noexcept {
  return std::get<0>(exprs);
}

template <typename TagType,
          TagType Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
inline decltype(auto) generic_op<TagType,
                                 Tag,
                                 ResultExprType,
                                 ResultType,
                                 NParameters,
                                 OperandTypes...>::second() const noexcept {
  return std::get<1>(exprs);
}

template <typename TagType,
          TagType Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
inline void generic_op<TagType,
                       Tag,
                       ResultExprType,
                       ResultType,
                       NParameters,
                       OperandTypes...>::traverse() const {
  auto& cache = thread_eval_cache;

  cache.cnt[this->id]++;

  // if cnt is bigger than 1, the subexpressions won't be evaluated
  if (cache.cnt[this->id] <= 1) {
    // traverse from NParameters to sizeof...(OperandTypes) - 1
    apply_void(take_rest<NParameters>(exprs),
               [&](auto const& expr) { mgcpp::traverse(expr); });
  }
}

namespace internal {
struct cache_lock_guard {
  cache_lock_guard() {
    auto& cache = thread_eval_cache;
    cache.total_computations = 0;
    cache.cache_hits = 0;
    cache.evaluating = true;
  }
  ~cache_lock_guard() {
    auto& cache = thread_eval_cache;
    cache.evaluating = false;
    MGCPP_ASSERT(cache.cnt.empty(),
                 "Cache counter is not empty after evaluation");
    MGCPP_ASSERT(cache.map.empty(), "Cache map is not empty after evaluation");
  }
};
}  // namespace internal

template <typename TagType,
          TagType Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
typename generic_op<TagType,
                    Tag,
                    ResultExprType,
                    ResultType,
                    NParameters,
                    OperandTypes...>::result_type
generic_op<TagType,
           Tag,
           ResultExprType,
           ResultType,
           NParameters,
           OperandTypes...>::eval() const {
  eval_context ctx;
  return this->eval(ctx);
}

template <typename TagType,
          TagType Tag,
          template <typename> class ResultExprType,
          typename ResultType,
          size_t NParameters,
          typename... OperandTypes>
typename generic_op<TagType,
                    Tag,
                    ResultExprType,
                    ResultType,
                    NParameters,
                    OperandTypes...>::result_type
generic_op<TagType,
           Tag,
           ResultExprType,
           ResultType,
           NParameters,
           OperandTypes...>::eval(eval_context& ctx) const {
  auto& cache = thread_eval_cache;

  // traverse the tree first to count the number of duplicate subtrees
  if (!cache.evaluating) {
    mgcpp::traverse(*this);

    internal::cache_lock_guard guard{};

    return this->eval(ctx);
  }

  cache.total_computations++;

  // try to find cache
  auto it = cache.map.find(this->id);

  // number of instances of this node left

  auto left = --cache.cnt.at(this->id);
  if (left == 0) {
    cache.cnt.erase(this->id);
  }

  // If cached, return the cache
  if (it != cache.map.end()) {
    cache.cache_hits++;
    auto cached = it->second.template get<result_type>();

    // Erase the cache for memory if it is no longer needed
    if (left == 0) {
      cache.map.erase(it);
    }

    return cached;
  }

  // If the same subexpression is shared by more than 1 nodes
  // and this is not a terminal node, cache
  if (!is_terminal && left >= 1) {
    auto result = evaluator::eval(*this, ctx);
    cache.map[this->id] = result;
    return result;
  }

  // No need to cache if the expression is not shared
  return evaluator::eval(*this, ctx);
}

}  // namespace mgcpp
