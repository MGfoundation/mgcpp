#include <atomic>
#include <mgcpp/expressions/expression.hpp>

namespace mgcpp {
expr_id_type make_id() {
  static std::atomic<expr_id_type> counter(0);
  return counter.fetch_add(1);
}
}  // namespace mgcpp
