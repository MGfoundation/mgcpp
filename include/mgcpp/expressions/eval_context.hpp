
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef EVAL_CONTEXT_HPP
#define EVAL_CONTEXT_HPP

#include <memory>
#include <mgcpp/expressions/expression.hpp>
#include <mgcpp/expressions/generic_op.hpp>
#include <tuple>
#include <type_traits>
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
    erased(T&& data)
        : m(std::make_shared<model<std::remove_reference_t<T>>>(
              std::forward<T>(data))) {}

    template <typename T>
    T get() const {
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

  std::unordered_map<int, erased> placeholders;

  template <int Num,
            template <typename> class ResultExprType,
            typename ResultType>
  void feed(placeholder_node<Num, ResultExprType, ResultType>,
            ResultType const& val) {
    placeholders[Num] = erased(val);
  }

  template <size_t PlaceholderID, typename ResultType>
  auto get_placeholder() const {
    return placeholders.at(PlaceholderID).get<ResultType>();
  }
};
}  // namespace mgcpp

#endif  // EVAL_CONTEXT_HPP
