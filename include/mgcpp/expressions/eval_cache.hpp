#ifndef EVAL_CACHE_HPP
#define EVAL_CACHE_HPP

#include <mgcpp/expressions/expression.hpp>
#include <mgcpp/expressions/generic_op.hpp>
#include <mgcpp/global/type_erased.hpp>
#include <unordered_map>

namespace mgcpp {

struct eval_cache {
  int total_computations = 0;
  int cache_hits = 0;
  bool evaluating = false;
  std::unordered_map<expr_id_type, int> cnt;
  std::unordered_map<expr_id_type, type_erased> map;
};

eval_cache& get_eval_cache();
}  // namespace mgcpp

#endif  // EVAL_CACHE_HPP
