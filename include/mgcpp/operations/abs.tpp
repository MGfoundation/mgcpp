
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/device/matrix.hpp>
#include <mgcpp/device/vector.hpp>
#include <mgcpp/kernels/mgblas_lv1.hpp>
#include <mgcpp/operations/abs.hpp>
#include <mgcpp/system/exception.hpp>

#include <cstdlib>

namespace mgcpp
{
    template<typename T,
             size_t Device,
             allignment Allign,
             typename Alloc>
    device_vector<T, Device, Allign, Alloc>
    strict::
    abs(device_vector<T, Device, Allign, Alloc> const& vec)
    {
        auto set_device_status = cuda_set_device(Device);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        size_t n = vec.shape();

        auto result = mgcpp::device_vector<float>(vec);
        auto status = mgblas_vab(result.data_mutable(), n);

        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }

    template<typename T,
             size_t Device,
             storage_order SO,
             typename Alloc>
    device_matrix<T, Device, SO, Alloc>
    strict::
    abs(device_matrix<T, Device, SO, Alloc> const& mat)
    {
        auto set_device_status = cuda_set_device(Device);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        auto shape = mat.shape();

        auto result = mgcpp::device_matrix<float>(mat);
        auto status = mgblas_vab(result.data_mutable(), shape.first * shape.second);

        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }
}

