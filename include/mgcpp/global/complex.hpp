
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
    union complex;

    template<>
    union complex<float>
    {
        cuComplex _device;
        std::complex<float> _host;

        inline complex<float>(std::complex<float> const& init)
            : _host(init) {}

        inline complex<float>(cuComplex const& init)
            : _device(init) {}

        inline complex<float>(std::complex<float>&& init)
            : _host(std::move(init)) {}

        inline complex<float>(cuComplex&& init)
            : _device(std::move(init)) {}

        inline complex<float>&
        operator=(std::complex<float> const& init)
        {
            _host = init;
            return *this;
        }

        inline complex<float>&
        operator=(cuComplex init)
        {
            _device = init;
            return *this;
        }

        inline complex<float>&
        operator=(std::complex<float>&& init)
        {
            _host = std::move(init);
            return *this;
        }

        inline complex<float>&
        operator=(cuComplex&& init)
        {
            _device = std::move(init);
            return *this;
        }

        operator cuComplex() const
        { return _device; }

        operator std::complex<float>() const
        { return _host; }
    };

    template<>
    union complex<double>
    {
        cuDoubleComplex _device;
        std::complex<double> _host;

        inline complex<double>(std::complex<double> const& init)
            : _host(init) {}

        inline complex<double>(cuDoubleComplex const& init)
            : _device(init) {}

        inline complex<double>(std::complex<double>&& init)
            : _host(std::move(init)) {}

        inline complex<double>(cuDoubleComplex&& init)
            : _device(std::move(init)) {}

        inline complex<double>&
        operator=(std::complex<double> const& init)
        {
            _host = init;
            return *this;
        }

        inline complex<double>&
        operator=(cuDoubleComplex init)
        {
            _device = init;
            return *this;
        }

        inline complex<double>&
        operator=(std::complex<double>&& init)
        {
            _host = std::move(init);
            return *this;
        }

        inline complex<double>&
        operator=(cuDoubleComplex&& init)
        {
            _device = std::move(init);
            return *this;
        }

        operator cuDoubleComplex() const
        { return _device; }

        operator std::complex<double>() const
        { return _host; }
    };
}

#endif
