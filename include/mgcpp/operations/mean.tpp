
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda/device.hpp>
#include <mgcpp/kernels/mgblas_lv1.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    template<typename DenseVec,
             typename Type,
             alignment Align,
             size_t DeviceId>
    Type
    strict::
    mean(dense_vector<DenseVec, Type, Align, DeviceId> const& vec)
    {
        auto const& original_vec = ~vec; 

        auto set_device_status = cuda_set_device(DeviceId);
        if(!set_device_status)
        { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

        size_t size = original_vec.shape();
        Type result;

        auto status = mgblas_vpr(original_vec.data(), &result, size);
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

        size_t total_size = shape.first * shape.second;
        Type result;
            
        auto status = mgblas_vpr(original_mat.data(),
                                 &result,
                                 total_size);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result / total_size;
    }
}
