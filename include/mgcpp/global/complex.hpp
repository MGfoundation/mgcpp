
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
        complex() = default;

        complex(std::complex<float> x)
            : real(x.real()), imag(x.imag())
        {}

        operator std::complex<float>() const
        {
            return {real, imag};
        }

    private:
        float real, imag;
    };

    template<>
    struct complex<double>
    {
        complex() = default;

        complex(std::complex<double> x)
            : real(x.real()), imag(x.imag())
        {}

        operator std::complex<double>() const
        {
            return {real, imag};
        }

    private:
        double real, imag;
    };
}

#endif
