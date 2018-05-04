#ifndef TUPLE_UTILS_HPP
#define TUPLE_UTILS_HPP

#include <tuple>
#include <type_traits>
#include <utility>

namespace mgcpp {
template <class Tuple, class F, size_t... Is>
constexpr decltype(auto) apply_impl(Tuple&& t,
                                    F&& f,
                                    std::index_sequence<Is...>) {
  return std::make_tuple(f(std::get<Is>(std::forward<Tuple>(t)))...);
}

template <class Tuple, class F>
constexpr decltype(auto) apply(Tuple&& t, F&& f) {
  return apply_impl(
      std::forward<Tuple>(t), std::forward<F>(f),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template <class Tuple, class F, size_t... Is>
constexpr void apply_void_impl(Tuple&& t, F&& f, std::index_sequence<Is...>) {
  (void)std::initializer_list<int>{
      (f(std::get<Is>(std::forward<Tuple>(t))), 0)...};
}

template <class Tuple, class F>
constexpr void apply_void(Tuple&& t, F&& f) {
  apply_void_impl(
      std::forward<Tuple>(t), std::forward<F>(f),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template <class Tuple, size_t... Is>
constexpr decltype(auto) take_front_impl(Tuple&& t,
                                         std::index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(std::forward<Tuple>(t))...);
}

template <size_t N, class Tuple>
constexpr decltype(auto) take_front(Tuple&& t) {
  return take_front_impl(std::forward<Tuple>(t), std::make_index_sequence<N>{});
}

template <size_t N, class Tuple, size_t... Is>
constexpr decltype(auto) take_rest_impl(Tuple&& t, std::index_sequence<Is...>) {
  return std::make_tuple(std::get<N + Is>(std::forward<Tuple>(t))...);
}

template <size_t N, class Tuple>
constexpr decltype(auto) take_rest(Tuple&& t) {
  return take_rest_impl<N>(
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value - N>{});
}
}  // namespace mgcpp
#endif  // TUPLE_UTILS_HPP
