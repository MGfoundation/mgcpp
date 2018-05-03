#ifndef TUPLE_UTILS_HPP
#define TUPLE_UTILS_HPP

#include <tuple>
#include <type_traits>
#include <utility>

namespace mgcpp {
template <class Tuple, class F, size_t... Is>
constexpr auto apply_impl(Tuple t, F f, std::index_sequence<Is...>) {
  return std::make_tuple(f(std::get<Is>(t))...);
}

template <class Tuple, class F>
constexpr auto apply(Tuple t, F f) {
  return apply_impl(t, f, std::make_index_sequence<std::tuple_size<Tuple>{}>{});
}

template <class Tuple, class F, size_t... Is>
constexpr void apply_void_impl(Tuple t, F f, std::index_sequence<Is...>) {
  (void)std::initializer_list<int>{(f(std::get<Is>(t)), 0)...};
}

template <class Tuple, class F>
constexpr void apply_void(Tuple t, F f) {
  apply_void_impl(t, f, std::make_index_sequence<std::tuple_size<Tuple>{}>{});
}

template <class Tuple, size_t... Is>
constexpr auto take_front_impl(Tuple t, std::index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(t)...);
}

template <size_t N, class Tuple>
constexpr auto take_front(Tuple t) {
  return take_front_impl(t, std::make_index_sequence<N>{}());
}

template <size_t N, class Tuple, size_t... Is>
constexpr auto take_rest_impl(Tuple t, std::index_sequence<Is...>) {
  return std::make_tuple(std::get<N + Is>(t)...);
}

template <size_t N, class Tuple>
constexpr auto take_rest(Tuple t) {
  return take_rest_impl<N>(
      t, std::make_index_sequence<std::tuple_size<Tuple>::value - N>{});
}
}  // namespace mgcpp
#endif  // TUPLE_UTILS_HPP
