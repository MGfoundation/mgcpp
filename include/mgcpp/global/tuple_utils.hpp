#ifndef TUPLE_UTILS_HPP
#define TUPLE_UTILS_HPP

#include <tuple>
#include <type_traits>
#include <utility>

namespace mgcpp {

namespace internal {
template <class Tuple, class F, size_t... Is>
constexpr decltype(auto) apply_impl(Tuple&& t,
                                    F&& f,
                                    std::index_sequence<Is...>) {
  return std::make_tuple(f(std::get<Is>(std::forward<Tuple>(t)))...);
}
}  // namespace internal

template <class Tuple, class F>
constexpr decltype(auto) apply(Tuple&& t, F&& f) {
  constexpr auto len = std::tuple_size<std::remove_reference_t<Tuple>>::value;
  return internal::apply_impl(std::forward<Tuple>(t), std::forward<F>(f),
                              std::make_index_sequence<len>{});
}

namespace internal {
template <class Tuple, class F, size_t... Is>
constexpr void apply_void_impl(Tuple&& t, F&& f, std::index_sequence<Is...>) {
  (void)std::initializer_list<int>{
      (f(std::get<Is>(std::forward<Tuple>(t))), 0)...};
}
}  // namespace internal

template <class Tuple, class F>
constexpr void apply_void(Tuple&& t, F&& f) {
  constexpr auto len = std::tuple_size<std::remove_reference_t<Tuple>>::value;
  internal::apply_void_impl(std::forward<Tuple>(t), std::forward<F>(f),
                            std::make_index_sequence<len>{});
}

namespace internal {
template <class Tuple, size_t... Is>
constexpr decltype(auto) take_front_impl(Tuple&& t,
                                         std::index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(std::forward<Tuple>(t))...);
}
}  // namespace internal

template <size_t N, class Tuple>
constexpr decltype(auto) take_front(Tuple&& t) {
  return internal::take_front_impl(std::forward<Tuple>(t),
                                   std::make_index_sequence<N>{});
}

namespace internal {
template <size_t N, class Tuple, size_t... Is>
constexpr decltype(auto) take_rest_impl(Tuple&& t, std::index_sequence<Is...>) {
  return std::make_tuple(std::get<N + Is>(std::forward<Tuple>(t))...);
}
}  // namespace internal

template <size_t N, class Tuple>
constexpr decltype(auto) take_rest(Tuple&& t) {
  constexpr auto len = std::tuple_size<std::remove_reference_t<Tuple>>::value;
  return internal::take_rest_impl<N>(std::forward<Tuple>(t),
                                     std::make_index_sequence<len - N>{});
}

namespace internal {
template <class Tuple, size_t I>
constexpr decltype(auto) sum_tuple_impl(Tuple&& t, std::index_sequence<I>) {
  return std::get<0>(std::forward<Tuple>(t));
}

template <class Tuple, size_t... Is>
constexpr decltype(auto) sum_tuple_impl(Tuple&& t, std::index_sequence<Is...>) {
  constexpr auto len = std::tuple_size<std::remove_reference_t<Tuple>>::value;
  return std::get<0>(std::forward<Tuple>(t)) +
         sum_tuple_impl(take_rest<1>(std::forward<Tuple>(t)),
                        std::make_index_sequence<len - 1>{});
}
}  // namespace internal

template <class Tuple>
constexpr decltype(auto) sum_tuple(Tuple&& t) {
  constexpr auto len = std::tuple_size<std::remove_reference_t<Tuple>>::value;
  return internal::sum_tuple_impl(std::forward<Tuple>(t),
                                  std::make_index_sequence<len>{});
}

namespace internal {
template <class Tuple1, class Tuple2, size_t... Is>
constexpr decltype(auto) zip_impl(Tuple1&& t1,
                                  Tuple2&& t2,
                                  std::index_sequence<Is...>) {
  return std::make_tuple(
      std::make_pair(std::get<Is>(std::forward<Tuple1>(t1)),
                     std::get<Is>(std::forward<Tuple2>(t2)))...);
}
}  // namespace internal

template <class Tuple1, class Tuple2>
constexpr decltype(auto) zip(Tuple1&& t1, Tuple2&& t2) {
  constexpr auto len1 = std::tuple_size<std::remove_reference_t<Tuple1>>::value;
  constexpr auto len2 = std::tuple_size<std::remove_reference_t<Tuple2>>::value;
  static_assert(len1 == len2, "Two tuple lengths must be the same");
  return internal::zip_impl(std::forward<Tuple1>(t1), std::forward<Tuple2>(t2),
                            std::make_index_sequence<len1>{});
}

namespace internal {
template <class Tuple, size_t... Is>
constexpr decltype(auto) enumerate_impl(Tuple&& t, std::index_sequence<Is...>) {
  return std::make_tuple(
      std::make_pair(Is, std::get<Is>(std::forward<Tuple>(t)))...);
}
}  // namespace internal

template <class Tuple>
constexpr decltype(auto) enumerate(Tuple&& t) {
  constexpr auto len = std::tuple_size<std::remove_reference_t<Tuple>>::value;
  return internal::enumerate_impl(std::forward<Tuple>(t),
                                  std::make_index_sequence<len>{});
}
}  // namespace mgcpp
#endif  // TUPLE_UTILS_HPP
