#include <mgcpp/expressions/scalar_expr.hpp>

namespace mgcpp {

template <typename Type>
scalar_constant_expr<Type> scal(Type scalar) {
  static_assert (mgcpp::is_scalar<Type>::value, "Type is not scalar");
  return scalar_constant_expr<Type>(scalar);
}

}  // namespace mgcpp
