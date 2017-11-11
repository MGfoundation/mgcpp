
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
    sum(device_vector<T, DeviceId, Allign, Alloc> const& vec)
    {
        auto set_device_status = cuda_set_device(DeviceId);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        T result;
        size_t size = vec.shape();
            
        auto status = mgblas_vpr(vec.data(), &result, size);

        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }

    template<typename DeviceMatrix,
             MGCPP_CONCEPT(is_device_matrix<DeviceMatrix>::value)>
    inline typename DeviceMatrix::value_type
    strict::
    sum(DeviceMatrix const& mat)
    {
        using value_type = typename DeviceMatrix::value_type;
        size_t const device_id = DeviceMatrix::device_id;

        auto set_device_status = cuda_set_device(device_id);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        value_type result;
        auto shape = mat.shape();
            
        auto status =
            mgblas_vpr(mat.data(), &result, shape.first * shape.second);

        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }
}
