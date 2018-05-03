
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_EXPRESSION_HPP_
#define _MGCPP_EXPRESSIONS_EXPRESSION_HPP_

#include <cstddef>
#include <utility>
#include <functional>

namespace mgcpp {

using expr_id_type = unsigned long;
expr_id_type make_id();

template <typename Type>
struct expression {
  inline Type& operator~() noexcept { return *static_cast<Type*>(this); }

  inline Type const& operator~() const noexcept {
    return *static_cast<Type const*>(this);
  }

protected:
  expr_id_type id = make_id();
};

struct eval_context;

template <typename T>
inline typename T::result_type eval(expression<T> const& expr,
                                    eval_context& ctx);

template <typename T>
inline typename T::result_type eval(expression<T> const& expr);


/*
template <typename... Ts>
inline std::tuple<typename Ts::result_type...> eval(
    std::tuple<Ts...> const& tuple,
    eval_context& ctx) {
  return std::make_tuple(eval(tuple, ctx)...);
}
*/

}  // namespace mgcpp

#endif
