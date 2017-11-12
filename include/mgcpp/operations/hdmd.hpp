
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_HADAMARD_HPP_
#define _MGCPP_OPERATIONS_HADAMARD_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <cstdlib>

namespace mgcpp
{
    namespace strict
    {
        template<typename LhsDenseVec,
                 typename RhsDenseVec,
                 typename Type,
                 allignment Allign,
                 size_t DeviceId>
        inline device_vector<Type, Allign, DeviceId,
                             typename LhsDenseVec::allocator_type>
        hdmd(dense_vector<LhsDenseVec, Type, Allign, DeviceId> const& lhs,
             dense_vector<RhsDenseVec, Type, Allign, DeviceId> const& rhs);
    }
}

#include <mgcpp/operations/hdmd.tpp>
#endif
