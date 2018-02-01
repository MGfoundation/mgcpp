
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_TYPE_TRAITS_HPP_
#define _MGCPP_TYPE_TRAITS_TYPE_TRAITS_HPP_

#include <type_traits>
#include <mgcpp/global/complex.hpp>
#include <mgcpp/global/half_precision.hpp>

namespace mgcpp
{
    template<typename... Args>
    using void_t = void;

    template<typename T>
    struct is_scalar
    {
        static const bool value = std::is_same<T, float>::value  ||
                                  std::is_same<T, double>::value ||
                                  std::is_same<T, complex<float>>::value  ||
                                  std::is_same<T, complex<double>>::value ||
                                  std::is_same<T, half>::value ||
                                  std::is_same<T, complex<half>>::value;
    };
}

#endif
