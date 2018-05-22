#ifndef EVAL_CACHE_HPP
#define EVAL_CACHE_HPP

#include <functional>
#include <mgcpp/global/type_erased.hpp>

namespace mgcpp {

size_t get_last_run_cache_hits();

void analyze_graph(size_t id, std::function<void()> traverse);

static_any evaluate_if_needed(size_t id,
                              bool needs_caching,
                              std::function<void()> traverse,
                              std::function<static_any()> evaluate);

}  // namespace mgcpp

#endif  // EVAL_CACHE_HPP
