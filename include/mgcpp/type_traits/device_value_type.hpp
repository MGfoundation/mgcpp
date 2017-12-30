
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_TRAITS_DEVICE_VALUE_TYPE_HPP_
#define _MGCPP_TYPE_TRAITS_DEVICE_VALUE_TYPE_HPP_

#include <mgcpp/global/complex.hpp>
#include <mgcpp/global/half_precision.hpp>
#include <cuComplex.h>
#include <cuda_fp16.h>

#include <complex>

namespace mgcpp
{
    template<typename Type>
    struct device_value_type
    { using type = Type; };

    template<>
    struct device_value_type<complex<float>>
    { using type = cuComplex; };

    template<>
    struct device_value_type<complex<double>>
    { using type = cuDoubleComplex; };

    template<>
    struct device_value_type<half>
    { using type = __half; };
}

#endif
