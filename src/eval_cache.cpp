#include <mgcpp/expressions/eval_cache.hpp>
#include <mgcpp/global/type_erased.hpp>

namespace mgcpp {
eval_cache& get_eval_cache() {
    static thread_local eval_cache cache{};
    return cache;
}

}  // namespace mgcpp
