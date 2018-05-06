#include <atomic>
#include <mgcpp/expressions/expression.hpp>

namespace mgcpp {
size_t make_id() {
  static std::atomic<size_t> counter(0);
  return counter.fetch_add(1);
}
}  // namespace mgcpp
