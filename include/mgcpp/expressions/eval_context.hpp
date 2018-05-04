
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef EVAL_CONTEXT_HPP
#define EVAL_CONTEXT_HPP

#include <memory>
#include <mgcpp/expressions/expression.hpp>
#include <mgcpp/expressions/generic_op.hpp>
#include <type_traits>
#include <unordered_map>

namespace mgcpp {

struct type_erased {
  type_erased() = default;

  template <typename T>
  type_erased(T&& data);

  template <typename T>
  T get() const;

  struct concept {
    virtual ~concept() = default;
  };

  template <typename T>
  struct model final : concept {
    model(T const& x);
    model(T&& x);
    T data;
  };

  std::shared_ptr<concept const> m;
};

struct eval_cache {
  int total_computations = 0;
  int cache_hits = 0;
  bool evaluating = false;
  std::unordered_map<expr_id_type, int> cnt;
  std::unordered_map<expr_id_type, type_erased> map;
};
extern thread_local eval_cache thread_eval_cache;

struct eval_context {
  /** Associate a value to a placeholder. The associated value will be fed to
   * the placeholder's place when the graph is evaluated.
   * \param ph The placeholder.
   * \param val The value associated with the placeholder.
   */
  template <int Num,
            template <typename> class ResultExprType,
            typename ResultType>
  void feed(placeholder_node<Num, ResultExprType, ResultType> ph,
            ResultType const& val);

  /** Get the value of the placeholder associated with the PlaceholderID.
   * ResultType should be the type of the value when feed() was called.
   */
  template <size_t PlaceholderID, typename ResultType>
  auto get_placeholder() const;

protected:
  std::unordered_map<int, type_erased> _placeholders;
};

}  // namespace mgcpp

#include <mgcpp/expressions/eval_context.tpp>

#endif  // EVAL_CONTEXT_HPP
