#include <mgcpp/expressions/eval_cache.hpp>
#include <mgcpp/global/type_erased.hpp>
#include <mgcpp/system/assert.hpp>
#include <unordered_map>

namespace mgcpp {

struct eval_cache {
  size_t total_computations = 0;
  size_t cache_hits = 0;
  bool evaluating = false;
  std::unordered_map<size_t, int> cnt;
  std::unordered_map<size_t, static_any> map;
};

eval_cache& get_eval_cache() {
  static thread_local eval_cache cache{};
  return cache;
}

size_t get_last_run_cache_hits() {
  return get_eval_cache().cache_hits;
}

void analyze_graph(size_t id, std::function<void()> traverse) {
  auto& cache = get_eval_cache();

  cache.cnt[id]++;

  // if cnt is bigger than 1, the subexpressions won't be evaluated
  if (cache.cnt[id] <= 1) {
    traverse();
  }
}

namespace {
struct cache_lock_guard {
  cache_lock_guard() {
    auto& cache = get_eval_cache();
    cache.total_computations = 0;
    cache.cache_hits = 0;
    cache.evaluating = true;
  }
  ~cache_lock_guard() {
    auto& cache = get_eval_cache();
    cache.evaluating = false;
    MGCPP_ASSERT(cache.cnt.empty(),
                 "Cache counter is not empty after evaluation");
    MGCPP_ASSERT(cache.map.empty(), "Cache map is not empty after evaluation");
  }
};
}  // namespace

static_any evaluate_if_needed(size_t id,
                              bool needs_caching,
                              std::function<void()> traverse,
                              std::function<static_any()> evaluate) {
  auto& cache = get_eval_cache();

  // traverse the tree first to count the number of duplicate subtrees
  if (!cache.evaluating) {
    traverse();

    cache_lock_guard guard{};

    return evaluate_if_needed(id, needs_caching, std::move(traverse),
                              std::move(evaluate));
  }

  cache.total_computations++;

  // try to find cache
  auto it = cache.map.find(id);

  // number of instances of this node left
  auto left = --cache.cnt.at(id);
  if (left == 0) {
    cache.cnt.erase(id);
  }

  // If cached, return the cache
  if (it != cache.map.end()) {
    cache.cache_hits++;
    auto cached = it->second;

    // Erase the cache for memory if it is no longer needed
    if (left == 0) {
      cache.map.erase(it);
    }

    return cached;
  }

  // If the same subexpression is shared by more than 1 nodes
  // and this is not a terminal node, cache
  if (needs_caching && left >= 1) {
    return cache.map[id] = evaluate();
  }

  // No need to cache if the expression is not shared
  return evaluate();
}

}  // namespace mgcpp
