
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
#include <type_traits>
#include <cstring>

namespace mgcpp
{

    template<typename InputType, typename OutputType>
    inline OutputType*
    mgcpp_cast(InputType const* first, InputType const* last, OutputType* d_first)
    {
        static_assert(std::is_same<InputType, OutputType>::value ||
                      (std::is_same<InputType, std::complex<float>>::value && std::is_same<OutputType, cuComplex>::value) ||
                      (std::is_same<InputType, std::complex<double>>::value && std::is_same<OutputType, cuDoubleComplex>::value) ||
                      (std::is_same<InputType, cuComplex>::value && std::is_same<OutputType, std::complex<float>>::value) ||
                      (std::is_same<InputType, cuDoubleComplex>::value && std::is_same<OutputType, std::complex<double>>::value),
                      "Types cannot be converted.");

        std::memcpy(d_first, first, (last - first) * sizeof(InputType));
        return d_first + (last - first);
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
