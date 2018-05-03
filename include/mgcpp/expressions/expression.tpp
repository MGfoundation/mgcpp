#ifndef EXPRESSION_TPP
#define EXPRESSION_TPP

#include <mgcpp/expressions/expression.hpp>
#include <mgcpp/expressions/eval_context.hpp>

namespace mgcpp {
template <typename T>
inline typename T::result_type eval(expression<T> const& expr,
                                    eval_context& ctx) {

  // traverse the tree first to count the number of duplicate subtrees
  if (!ctx.is_evaluating) {
    traverse(expr, ctx);
  }

  // mask the context so that traverse() won't be called hereafter
  ctx.is_evaluating = true;
  return (~expr).eval(ctx);
}

template <typename T>
inline typename T::result_type eval(expression<T> const& expr) {
  eval_context ctx;
  return eval(expr, ctx);
}

template <typename T>
inline void traverse(expression<T> const& expr, eval_context& ctx) {
  (~expr).traverse(ctx);
}
}

#endif // EXPRESSION_TPP
