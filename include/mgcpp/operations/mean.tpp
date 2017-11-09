
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cstdlib>
#include <algorithm>

#include <mgcpp/cublas/blas_lv1.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/device/matrix.hpp>
#include <mgcpp/device/vector.hpp>
#include <mgcpp/kernels/mgblas_helpers.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    namespace strict
    {
        template<typename T,
                 size_t DeviceId,
                 allignment Allign,
                 typename Alloc>
        T
        mean(device_vector<T, DeviceId, Allign, Alloc> const& vec)
        {
            auto set_device_status = cuda_set_device(DeviceId);
            if(!set_device_status)
            { MGCPP_THROW_SYSTEM_ERROR(set_device_status.error()); }

            T result;
            size_t size = vec.shape();
            
            auto temp = cuda_malloc<T>(size);
            if(!temp)
            { MGCPP_THROW_SYSTEM_ERROR(temp.error()); }

            auto fill_status = mgblas_fill(temp.value(), 1, size);
            if(!fill_status)
            { MGCPP_THROW_SYSTEM_ERROR(fill_status.error()); }

            auto* context = vec.context();
            auto handle = context->get_cublas_context(DeviceId);
            
            auto status = cublas_dot(handle, size,
                                     temp.value(), 1,
                                     vec.data(), 1,
                                     &result);

            (void)cuda_free(temp.value());
            if(!status)
            { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

            return result / size;
        }
    }
}
