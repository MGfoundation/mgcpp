
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef EVAL_CONTEXT_HPP
#define EVAL_CONTEXT_HPP

#include <memory>
#include <mgcpp/expressions/expression.hpp>
#include <unordered_map>
#include <utility>

namespace mgcpp {

struct eval_context {
  int total_computations = 0;
  int cache_hits = 0;
  bool is_evaluating = false;

  struct erased {
    erased() = default;

    template <typename T>
    erased(T const& data) : m(std::make_shared<model<T>>(data)) {}

    template <typename T>
    T get() {
      return static_cast<model<T> const&>(*m).data;
    }

    struct concept {
      virtual ~concept() = default;
    };
    template <typename T>
    struct model final : concept {
      model(T const& x) : data(x) {}
      model(T&& x) : data(std::move(x)) {}
      T data;
    };

    std::shared_ptr<concept const> m;
  };

  std::unordered_map<expr_id_type, size_t> cnt;
  std::unordered_map<expr_id_type, erased> cache;
};
}  // namespace mgcpp

#endif  // EVAL_CONTEXT_HPP
