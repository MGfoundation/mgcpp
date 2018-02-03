
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GLOBAL_HALF_PRECISION_HPP_
#define _MGCPP_GLOBAL_HALF_PRECISION_HPP_

#ifdef USE_HALF
#include <half/half.hpp>
#include <cuda_fp16.h>
namespace mgcpp
{
    using half = half_float::half;
}
#endif

#endif
