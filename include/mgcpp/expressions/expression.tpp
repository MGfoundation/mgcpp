#ifndef EXPRESSION_TPP
#define EXPRESSION_TPP

#include <mgcpp/expressions/expression.hpp>
#include <mgcpp/expressions/eval_context.hpp>

namespace mgcpp {
template <typename T>
inline typename T::result_type eval(expression<T> const& expr,
                                    eval_context& ctx) {
  //(~expr).traverse(ctx);
  return (~expr).eval(ctx);
}

template <typename T>
inline typename T::result_type eval(expression<T> const& expr) {
  eval_context ctx;
  return eval(expr, ctx);
}
}

#endif // EXPRESSION_TPP
