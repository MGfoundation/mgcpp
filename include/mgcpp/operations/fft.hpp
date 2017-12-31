
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_FFT_HPP_
#define _MGCPP_OPERATIONS_FFT_HPP_

#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>
#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/global/complex.hpp>

namespace mgcpp
{
    enum class fft_direction {
        forward, inverse
    };

    namespace strict
    {
        template<typename DeviceVec,
                 typename Type,
                 alignment Align,
                 size_t DeviceId>
        inline decltype(auto)
        rfft(dense_vector<DeviceVec, Type, Align, DeviceId> const& vec);

        template<typename DeviceVec,
                 typename Type,
                 alignment Align,
                 size_t DeviceId>
        inline decltype(auto)
        irfft(dense_vector<DeviceVec, complex<Type>, Align, DeviceId> const& vec,
              int n = -1);

        template<typename DeviceVec,
                 typename Type,
                 alignment Align,
                 size_t DeviceId>
        inline decltype(auto)
        cfft(dense_vector<DeviceVec, complex<Type>, Align, DeviceId> const& vec,
             fft_direction direction);

        template<typename DeviceMat,
                 typename Type,
                 size_t DeviceId>
        inline decltype(auto)
        rfft(dense_matrix<DeviceMat, Type, DeviceId> const& mat);

        template<typename DeviceMat,
                 typename Type,
                 size_t DeviceId>
        inline decltype(auto)
        irfft(dense_matrix<DeviceMat, Type, DeviceId> const& vec,
              int n = -1);

        template<typename DeviceMat,
                 typename Type,
                 size_t DeviceId>
        inline decltype(auto)
        cfft(dense_matrix<DeviceMat, Type, DeviceId> const& vec,
             fft_direction direction);
    }
}

#include <mgcpp/operations/fft.tpp>

#endif
