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

template <typename Type>
void expression<Type>::traverse() const {
  analyze_graph(id, [&] {
    apply_void((~*this).operands(),
               [](auto const& ch) { ch.traverse(); });
  });
}

template <typename T>
inline typename T::result_type eval(expression<T> const& expr,
                                    eval_context const& ctx) {
  return (~expr).eval(ctx);
}

template <typename T>
inline typename T::result_type eval(expression<T> const& expr) {
  return (~expr).eval();
}

}  // namespace mgcpp
