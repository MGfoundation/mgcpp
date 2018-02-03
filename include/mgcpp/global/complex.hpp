
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TYPE_COMPLEX_HPP_
#define _MGCPP_TYPE_COMPLEX_HPP_

#include <cuComplex.h>
#include <complex>

namespace mgcpp
{
    template<typename Type>
    using complex = std::complex<Type>;
}

#endif
