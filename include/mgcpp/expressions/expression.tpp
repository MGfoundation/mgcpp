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

}  // namespace mgcpp
