
#ifndef _MGCPP_OPERATIONS_FFT_HPP_
#define _MGCPP_OPERATIONS_FFT_HPP_

#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>
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
                size_t DeviceId,
                typename Alloc = mgcpp::default_allocator<complex<Type>, DeviceId>>
        inline device_vector<complex<Type>, Align, DeviceId>
        rfft(dense_vector<DeviceVec, Type, Align, DeviceId> const& vec);

        template<typename DeviceVec,
                typename Type,
                alignment Align,
                size_t DeviceId,
                typename Alloc = mgcpp::default_allocator<Type, DeviceId>>
        inline device_vector<Type, Align, DeviceId>
        irfft(dense_vector<DeviceVec, complex<Type>, Align, DeviceId> const& vec, int n = -1);

        template<typename DeviceVec,
                typename Type,
                alignment Align,
                size_t DeviceId>
        inline device_vector<complex<Type>, Align, DeviceId>
        cfft(dense_vector<DeviceVec, complex<Type>, Align, DeviceId> const& vec, fft_direction direction);
    }
}

#include <mgcpp/operations/fft.tpp>

#endif
