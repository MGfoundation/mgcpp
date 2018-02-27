
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_IS_SUPPORTED_TYPE_HPP_
#define _MGCPP_TYPE_TRAITS_IS_SUPPORTED_TYPE_HPP_

#include <complex>
#include <mgcpp/global/half_precision.hpp>
#include <type_traits>

namespace mgcpp {
template <typename T>
struct is_supported_type : std::false_type {};

template <>
struct is_supported_type<float> : std::true_type {};

template <>
struct is_supported_type<double> : std::true_type {};

template <>
struct is_supported_type<std::complex<float>> : std::true_type {};

template <>
struct is_supported_type<std::complex<double>> : std::true_type {};

#ifdef USE_HALF
template <>
struct is_supported_type<half> : std::true_type {};
#endif
}  // namespace mgcpp
#endif
