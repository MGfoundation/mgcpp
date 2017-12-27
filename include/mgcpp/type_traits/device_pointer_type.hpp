
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_POINTER_TYPE_HPP_
#define _MGCPP_TYPE_TRAITS_POINTER_TYPE_HPP_

#include <mgcpp/global/complex.hpp>
#include <cuComplex.h>

namespace mgcpp
{
    template<typename Type>
    struct device_pointer
    { using type = Type*; };

    template<typename Type>
    struct const_device_pointer
    { using type = Type const*; };

    template<>
    struct device_pointer<complex<float>>
    { using type = cuComplex*; };

    template<>
    struct device_pointer<complex<double>>
    { using type = cuDoubleComplex*; };

    template<>
    struct const_device_pointer<complex<float>>
    { using type = cuComplex const*; };

    template<>
    struct const_device_pointer<complex<double>>
    { using type = cuDoubleComplex const*; };
}

#endif
