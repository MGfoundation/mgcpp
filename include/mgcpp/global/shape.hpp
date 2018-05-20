
#ifndef _MGCPP_GLOBAL_SHAPE_HPP_
#define _MGCPP_GLOBAL_SHAPE_HPP_

#include <tuple>
#include <type_traits>

namespace mgcpp {
template <size_t Dims>
struct shape {
  size_t dims[Dims];

  using shape_type = shape<Dims>;

  shape();
  shape(std::initializer_list<size_t> list);
  shape(shape const&) = default;
  shape& operator=(shape const&) = default;

  size_t operator[](size_t idx) const;
  size_t& operator[](size_t idx);
  bool operator==(shape const& rhs) const;
  bool operator!=(shape const& rhs) const;

  template <std::size_t N>
  size_t get() const;
};

template <typename... Types>
shape<sizeof...(Types)> make_shape(Types... args);
}  // namespace mgcpp

namespace std {
template <size_t Dims>
class tuple_size<mgcpp::shape<Dims>>
    : std::integral_constant<std::size_t, Dims> {};

template <std::size_t N, size_t Dims>
struct tuple_element<N, mgcpp::shape<Dims>> {
  using type = std::size_t;
};
}  // namespace std

#include <mgcpp/global/shape.tpp>

#endif
