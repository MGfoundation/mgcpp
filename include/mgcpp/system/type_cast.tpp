
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cuComplex.h>
#include <cuda_fp16.h>
#include <mgcpp/global/complex.hpp>
#include <mgcpp/global/half_precision.hpp>
#include <mgcpp/system/type_cast.hpp>

#include <complex>

namespace mgcpp
{
    template<typename InputType, typename OutputType>
    inline OutputType*
    mgcpp_cast(InputType const* first, InputType const* last, OutputType* d_first)
    {
        if (first == d_first)
            return d_first + (last - first);

        return std::copy(first, last, d_first);
    }

    template<>
    inline cuComplex*
    mgcpp_cast(std::complex<float> const* first, std::complex<float> const* last, cuComplex* d_first)
    {
        return std::copy(reinterpret_cast<cuComplex const*>(first),
                         reinterpret_cast<cuComplex const*>(last), d_first);
    }

    template<>
    inline std::complex<float>*
    mgcpp_cast(cuComplex const* first, cuComplex const* last, std::complex<float>* d_first)
    {
        return reinterpret_cast<std::complex<float>*>(
            std::copy(first, last, reinterpret_cast<cuComplex*>(d_first)));
    }

    template<>
    inline cuDoubleComplex*
    mgcpp_cast(std::complex<double> const* first, std::complex<double> const* last, cuDoubleComplex* d_first)
    {
        return std::copy(reinterpret_cast<cuDoubleComplex const*>(first),
                         reinterpret_cast<cuDoubleComplex const*>(last), d_first);
    }

    template<>
    inline std::complex<double>*
    mgcpp_cast(cuDoubleComplex const* first, cuDoubleComplex const* last, std::complex<double>* d_first)
    {
        return reinterpret_cast<std::complex<double>*>(
            std::copy(first, last, reinterpret_cast<cuDoubleComplex*>(d_first)));
    }

    void half_to_float_impl(__half const* first, __half const* last, float* d_first);
    void float_to_half_impl(float const* first, float const* last, __half* d_first);

    template<>
    inline float*
    mgcpp_cast(__half const* first, __half const* last, float* d_first)
    {
        half_to_float_impl(first, last, d_first);
        return d_first + (last - first);
    }

    template<>
    inline __half*
    mgcpp_cast(float const* first, float const* last, __half* d_first)
    {
        float_to_half_impl(first, last, d_first);
        return d_first + (last - first);
    }
}
