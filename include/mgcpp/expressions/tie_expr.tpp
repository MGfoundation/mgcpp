#include <mgcpp/expressions/tie_expr.hpp>

namespace mgcpp {

template <typename... Exprs>
inline tie_op<Exprs...> tie(Exprs const&... exprs) {
  return tie_op<Exprs...>(~exprs...);
}

}  // namespace mgcpp
