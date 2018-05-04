#ifndef EXPRESSION_TPP
#define EXPRESSION_TPP

#include <mgcpp/expressions/eval_context.hpp>
#include <mgcpp/expressions/expression.hpp>

namespace mgcpp {

template <typename Type>
Type& expression<Type>::operator~() noexcept {
  return *static_cast<Type*>(this);
}

template <typename Type>
const Type& expression<Type>::operator~() const noexcept {
  return *static_cast<Type const*>(this);
}

template <typename T>
inline typename T::result_type eval(expression<T> const& expr,
                                    eval_context& ctx) {
  return (~expr).eval(ctx);
}

template <typename T>
inline typename T::result_type eval(expression<T> const& expr) {
  eval_context ctx;
  return eval(expr, ctx);
}

template <typename T>
inline void traverse(expression<T> const& expr) {
  (~expr).traverse();
}

}  // namespace mgcpp

#endif  // EXPRESSION_TPP
