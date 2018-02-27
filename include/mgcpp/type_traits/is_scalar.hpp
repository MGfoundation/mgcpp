
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_IS_SCALAR_HPP_
#define _MGCPP_TYPE_TRAITS_IS_SCALAR_HPP_

#include <complex>
#include <mgcpp/global/half_precision.hpp>
#include <type_traits>

namespace mgcpp {
template <typename T>
struct is_scalar {
  static const bool value =
      std::is_arithmetic<T>::value && !std::is_same<T, bool>::value &&
      !std::is_same<T, char>::value && !std::is_same<T, char16_t>::value &&
      !std::is_same<T, char32_t>::value && !std::is_same<T, wchar_t>::value &&
      !std::is_same<T, signed char>::value;
};

template <typename T>
struct is_scalar<std::complex<T>> : std::true_type {};

#ifdef USE_HALF
template <>
struct is_scalar<half> : std::true_type {};
#endif
}  // namespace mgcpp

#endif
