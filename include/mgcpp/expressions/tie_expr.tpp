#include <mgcpp/expressions/tie_expr.hpp>

namespace mgcpp {

template <typename... Exprs>
inline symbolic_tie_expr<Exprs...> tie(Exprs const& ... exprs) {
  return symbolic_tie_expr<Exprs...>(~exprs...);
}

}
