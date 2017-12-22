
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
}

#include <mgcpp/operations/fft.tpp>

#endif
