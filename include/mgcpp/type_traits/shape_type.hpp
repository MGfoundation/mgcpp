#ifndef SHAPE_TYPE_HPP
#define SHAPE_TYPE_HPP

#include <mgcpp/global/shape.hpp>
#include <tuple>
#include <type_traits>

namespace mgcpp {

template <typename T>
struct has_shape {
 private:
  typedef char one;
  typedef struct {
    char array[2];
  } two;

  template <typename C>
  static one test(typename C::shape_type*);
  template <typename C>
  static two test(...);

 public:
  static const bool value = sizeof(test<T>(0)) == sizeof(one);
};

}  // namespace mgcpp
#endif  // SHAPE_TYPE_HPP
