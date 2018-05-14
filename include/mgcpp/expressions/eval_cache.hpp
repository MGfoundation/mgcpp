#ifndef EVAL_CACHE_HPP
#define EVAL_CACHE_HPP

#include <mgcpp/global/type_erased.hpp>
#include <unordered_map>

namespace mgcpp {

struct eval_cache {
  int total_computations = 0;
  int cache_hits = 0;
  bool evaluating = false;
  std::unordered_map<size_t, int> cnt;
  std::unordered_map<size_t, static_any> map;
};

eval_cache& get_eval_cache();
}  // namespace mgcpp

#endif  // EVAL_CACHE_HPP
