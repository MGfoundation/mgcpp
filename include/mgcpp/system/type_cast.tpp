
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
    template<typename OutputType, typename InputType>
    OutputType
    mgcpp_cast(InputType data)
    { return data; }

    template<>
    inline std::complex<float>*
    mgcpp_cast<>(cuComplex* data)
    {
        return reinterpret_cast<std::complex<float>*>(data);
    }

    template<>
    inline std::complex<double>*
    mgcpp_cast<>(cuDoubleComplex* data)
    {
        return reinterpret_cast<std::complex<double>*>(data);
    }

    template<>
    inline std::complex<float> const*
    mgcpp_cast<>(cuComplex const* data)
    {
        return reinterpret_cast<std::complex<float> const*>(data);
    }

    template<>
    inline std::complex<double> const*
    mgcpp_cast<>(cuDoubleComplex const* data)
    {
        return reinterpret_cast<std::complex<double> const*>(data);
    }

    template<>
    inline cuComplex*
    mgcpp_cast<>(std::complex<float>* data)
    {
        return reinterpret_cast<cuComplex*>(data);
    }

    template<>
    inline cuDoubleComplex*
    mgcpp_cast<>(std::complex<double>* data)
    {
        return reinterpret_cast<cuDoubleComplex*>(data);
    }

    template<>
    inline cuComplex const*
    mgcpp_cast<>(std::complex<float> const* data)
    {
        return reinterpret_cast<cuComplex const*>(data);
    }

    template<>
    inline cuDoubleComplex const*
    mgcpp_cast<>(std::complex<double> const* data)
    {
        return reinterpret_cast<cuDoubleComplex const*>(data);
    }
}
