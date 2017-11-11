
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cstdlib>
#include <algorithm>

#include <mgcpp/cuda/device.hpp>
#include <mgcpp/kernels/mgblas_lv1.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    template<typename T,
             size_t DeviceId,
             allignment Allign,
             typename Alloc>
    T
    strict::
    mean(device_vector<T, DeviceId, Allign, Alloc> const& vec)
    {
        auto set_device_status = cuda_set_device(DeviceId);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        T result;
        size_t size = vec.shape();
            
        auto status = mgblas_vpr(vec.data(), &result, size);

        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result / size;
    }

    template<typename DeviceMatrix,
             typename Type,
             size_t DeviceId>
    Type
    strict::
    mean(dense_matrix<DeviceMatrix, Type, DeviceId> const& mat)
    {
        auto const& original_mat = ~mat; 

        auto set_device_status = cuda_set_device(DeviceId);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        auto shape = original_mat.shape();
        Type result;
        size_t total_size = shape.first * shape.second;
            
        auto status = mgblas_vpr(original_mat.data(),
                                 &result,
                                 total_size);

        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result / total_size;
    }
}
