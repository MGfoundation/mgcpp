
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cstdlib>
#include <algorithm>

#include <mgcpp/gpu/vector.hpp>
#include <mgcpp/cublas/blas_lv1.hpp>
#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/cuda/device.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    namespace strict
    {
        template<typename T,
                 size_t DeviceId,
                 allignment Allign>
        T
        sum(device_vector<T, DeviceId, Allign> const& vec)
        {
            auto set_device_status = cuda_set_device(DeviceId);
            if(!set_device_status)
            {
                MGCPP_THROW_SYSTEM_ERROR(set_device_status.error());
            }

            T result;
            size_t size = vec.shape();
            
            T* buffer = (T*)malloc(sizeof(T) * size);
            if(!buffer)
            {
                MGCPP_THROW_BAD_ALLOC; 
            }

            std::fill(buffer, buffer + size, 1);

            auto temp = cuda_malloc<T>(size);
            if(!temp)
            {
                MGCPP_THROW_SYSTEM_ERROR(temp.error()); 
            }

            auto memcpy_status =
                cuda_memcpy(temp.value(), buffer, size,
                            cuda_memcpy_kind::host_to_device);
            if(!memcpy_status)
            {
                free(buffer);
                (void)cuda_free(temp.value());
                MGCPP_THROW_SYSTEM_ERROR(memcpy_status.error());
            }

            auto* context = vec.context();
            auto handle = context->get_cublas_context(DeviceId);
            
            auto status = cublas_dot(handle, size,
                                     temp.value(), 1,
                                     vec.data(), 1,
                                     &result);

            free(buffer);
            (void)cuda_free(temp.value());
            if(!status)
            {
                MGCPP_THROW_SYSTEM_ERROR(status.error());
            }

            return result;
        }
    }
}
