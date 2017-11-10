
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
    template<typename LhsMat, typename RhsMat, typename>
    device_matrix<typename LhsMat::value_type,
                  LhsMat::device_id,
                  typename LhsMat::allocator_type>
    strict::
    mult(LhsMat const& first, RhsMat const& second)
    {
        using value_type = typename LhsMat::value_type;
        using allocator_type = typename LhsMat::allocator_type;
        size_t const device_id = LhsMat::device_id; 

        MGCPP_ASSERT(first.shape().second == second.shape().first,
                     "matrix dimensions didn't match");

        value_type const alpha = 1;
        value_type const beta = 0;

        auto second_shape = second.shape();
        auto first_shape = first.shape();

        size_t m = first_shape.first;
        size_t k = first_shape.second;
        size_t n = second_shape.second;

        device_matrix<value_type,
                      device_id,
                      allocator_type> result{m, n};

        auto* context = first.context();
        auto handle = context->get_cublas_context(device_id);
    
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
