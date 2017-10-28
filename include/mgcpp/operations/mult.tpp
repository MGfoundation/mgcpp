
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/cublas/blas_lv1.hpp>
#include <mgcpp/cublas/blas_lv3.hpp>
#include <mgcpp/device/matrix.hpp>
#include <mgcpp/device/vector.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/system/assert.hpp>

namespace mgcpp
{
    template<typename T, size_t Device, storage_order SO, typename Alloc>
    device_matrix<T, Device, SO, Alloc>
    strict::
    mult(device_matrix<T, Device, SO, Alloc> const& first,
         device_matrix<T, Device, SO, Alloc> const& second)
    {
        MGCPP_ASSERT(
            first.shape().second == second.shape().first,
            "matrix dimensions didn't match");

        T const alpha = 1;
        T const beta = 0;

        auto second_shape = second.shape();
        auto first_shape = first.shape();

        size_t m = first_shape.first;
        size_t k = first_shape.second;
        size_t n = second_shape.second;

        device_matrix<T, Device, SO, Alloc> result{m, n};

        auto* context = first.context();
        auto handle = context->get_cublas_context(Device);
    
        auto status = cublas_gemm(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k,
                                  &alpha,
                                  first.data(), m,
                                  second.data(), k,
                                  &beta,
                                  result.data_mutable(), m);

        if(!status)
            MGCPP_THROW_SYSTEM_ERROR(status.error());

        return result;
    }
    
    template<typename T, size_t Device, allignment Allign, typename Alloc>
    device_vector<T, Device, Allign, Alloc>
    strict::
    mult(T scalar,
         device_vector<T, Device, Allign, Alloc> const& vec)
    {
        auto* context = vec.context();
        auto handle = context->get_cublas_context(Device);
        auto size = vec.shape();

        device_vector<T, Device, Allign, Alloc> result(vec);

        auto status = cublas_scal(handle, size,
                                  &scalar,
                                  result.data_mutable(), 1);
        if(!status)
        {
            MGCPP_THROW_SYSTEM_ERROR(status.error());
        }

        return result;
    }
}
