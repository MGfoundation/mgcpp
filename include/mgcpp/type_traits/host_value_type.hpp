
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_HOST_VALUE_TYPE_HPP_
#define _MGCPP_TYPE_TRAITS_HOST_VALUE_TYPE_HPP_

#include <mgcpp/global/complex.hpp>
#include <mgcpp/global/half_precision.hpp>

#include <complex>

namespace mgcpp
{
    template<typename Type>
    struct value_type
    { using type = Type; };

    template<typename Type>
    struct value_type<complex<Type>>
    { using type = std::complex<Type>; };

    template<>
    struct value_type<half>
    { using type = float; };
}

#endif
