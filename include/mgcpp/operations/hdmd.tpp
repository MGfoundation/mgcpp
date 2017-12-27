
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/mgblas_lv1.hpp>
#include <mgcpp/operations/hdmd.hpp>
#include <mgcpp/system/assert.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    template<typename LhsDenseVec,
             typename RhsDenseVec,
             typename Type,
             alignment Align,
             size_t DeviceId>
    device_vector<Type, Align, DeviceId, 
                  typename LhsDenseVec::allocator_type>
    strict::
    hdmd(dense_vector<LhsDenseVec, Type, Align, DeviceId> const& lhs,
         dense_vector<RhsDenseVec, Type, Align, DeviceId> const& rhs)
    {
        using allocator_type = typename LhsDenseVec::allocator_type;

        auto const& lhs_vec = ~lhs;
        auto const& rhs_vec = ~rhs;

        MGCPP_ASSERT(lhs_vec.shape() == rhs_vec.shape(),
                     "matrix dimensions didn't match");

        size_t size = lhs_vec.shape();

        auto result = device_vector<Type,
                                    Align,
                                    DeviceId,
                                    allocator_type>(size);
        auto status = mgblas_vhp(lhs_vec.data(), rhs_vec.data(),
                                 result.data_mutable(),
                                 size);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }
}
