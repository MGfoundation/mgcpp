
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_COMPLEX_HPP_
#define _MGCPP_TYPE_COMPLEX_HPP_

#include <complex>

namespace mgcpp
{
    template<typename Type>
    struct complex;

    template<>
    struct complex<float>
    {
        ~complex() = delete;
    };

    template<>
    struct complex<double>
    {
        ~complex() = delete;
    };
}

#endif
