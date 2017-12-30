
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cuComplex.h>
#include <mgcpp/global/complex.hpp>
#include <mgcpp/global/half_precision.hpp>
#include <mgcpp/system/type_cast.hpp>

#include <complex>

namespace mgcpp
{
    template<typename InputType, typename OutputType>
    inline void
    mgcpp_cast(InputType const* first, InputType const* last, OutputType* d_first)
    {
        if (first == d_first)
            return;

        std::copy(first, last, d_first);
    }

    template<>
    inline void
    mgcpp_cast(std::complex<float> const* first, std::complex<float> const* last, cuComplex* d_first)
    {
        std::copy(first, last, reinterpret_cast<std::complex<float>*>(d_first));
    }

    template<>
    inline void
    mgcpp_cast(cuComplex const* first, cuComplex const* last, std::complex<float>* d_first)
    {
        std::copy(first, last, reinterpret_cast<cuComplex*>(d_first));
    }

    template<>
    inline void
    mgcpp_cast(std::complex<double> const* first, std::complex<double> const* last, cuDoubleComplex* d_first)
    {
        std::copy(first, last, reinterpret_cast<std::complex<double>*>(d_first));
    }

    template<>
    inline void
    mgcpp_cast(cuDoubleComplex const* first, cuDoubleComplex const* last, std::complex<double>* d_first)
    {
        std::copy(first, last, reinterpret_cast<cuDoubleComplex*>(d_first));
    }
}
