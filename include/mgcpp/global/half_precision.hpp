
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
    union half
    {
        __half _device;
        half_float::half _host;

        inline half(half_float::half const& init)
            : _host(init) {}

        inline half(__half const& init)
            : _device(init) {}

        inline half(half_float::half&& init)
            : _host(std::move(init)) {}

        inline half(cuDoubleComplex&& init)
            : _device(std::move(init)) {}

        inline half&
        operator=(half_float::half const& init)
        {
            _host = init;
            return *this;
        }

        inline half&
        operator=(cuDoubleComplex init)
        {
            _device = init;
            return *this;
        }

        inline half&
        operator=(half_float::half&& init)
        {
            _host = std::move(init);
            return *this;
        }

        inline half&
        operator=(cuDoubleComplex&& init)
        {
            _device = std::move(init);
            return *this;
        }

        operator __half() const
        { return _device; }

        operator half_float::half() const
        { return _host; }
    };
}
#endif

#endif
