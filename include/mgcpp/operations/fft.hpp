
#ifndef _MGCPP_OPERATIONS_FFT_HPP_
#define _MGCPP_OPERATIONS_FFT_HPP_

#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

namespace mgcpp
{
    template<typename DeviceVec,
             typename Type,
             allignment Align,
             size_t DeviceId>
    inline device_vector<Type, Align, DeviceId,
                         typename DeviceVec::allocator_type>
    rfft(dense_vector<DeviceVec, Type, Align, DeviceId> const& vec);

    template<typename DeviceVec,
             typename Type,
             allignment Align,
             size_t DeviceId>
    inline device_vector<Type, Align, DeviceId,
                         typename DeviceVec::allocator_type>
    irfft(dense_vector<DeviceVec, Type, Align, DeviceId> const& vec, int n = -1);

    enum class fft_direction {
        forward, inverse
    };

    template<typename DeviceVec,
             typename Type,
             allignment Align,
             size_t DeviceId>
    inline device_vector<Type, Align, DeviceId,
                         typename DeviceVec::allocator_type>
    cfft(dense_vector<DeviceVec, Type, Align, DeviceId> const& vec, fft_direction direction);
}

#include <mgcpp/operations/fft.tpp>

#endif
