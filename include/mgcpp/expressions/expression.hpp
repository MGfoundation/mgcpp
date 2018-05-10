
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_EXPRESSION_HPP_
#define _MGCPP_EXPRESSIONS_EXPRESSION_HPP_

#include <cstddef>
#include <utility>
#include <functional>
#include <mgcpp/expressions/forward.hpp>
#include <mgcpp/expressions/placeholder.hpp>

namespace mgcpp {

size_t make_id();

template <typename Type>
struct expression {
  inline Type& operator~() noexcept;

  inline Type const& operator~() const noexcept;

protected:
  size_t id = make_id();
};

template <typename T>
inline typename T::result_type eval(expression<T> const& expr,
                                    eval_context& ctx);

template <typename T>
inline typename T::result_type eval(expression<T> const& expr);

template <typename T>
inline void traverse(expression<T> const& expr);

}  // namespace mgcpp

#include <mgcpp/expressions/expression.tpp>
#endif
