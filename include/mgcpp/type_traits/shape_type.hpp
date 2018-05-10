#ifndef SHAPE_TYPE_HPP
#define SHAPE_TYPE_HPP

#include <mgcpp/global/shape.hpp>
#include <tuple>
#include <type_traits>

namespace mgcpp {

template<typename T>
struct has_shape
{
private:
  typedef char                      one;
  typedef struct { char array[2]; } two;

  template<typename C> static one test(typename C::shape_type*);
  template<typename C> static two test(...);
public:
  static const bool value = sizeof(test<T>(0)) == sizeof(one);
};

template<bool B, typename T>
struct shape_type_impl {
    using type = shape<0>;
};

template<typename T>
struct shape_type_impl<true, T> {
    using type = typename T::shape_type;
};

template <typename T, typename = void>
struct shape_type {
  using type = typename shape_type_impl<has_shape<T>::value, T>::type;
};

template <typename... Items>
struct shape_type<std::tuple<Items...>> {
  using type = std::tuple<shape_type<Items>...>;
};

}  // namespace mgcpp
#endif  // SHAPE_TYPE_HPP
