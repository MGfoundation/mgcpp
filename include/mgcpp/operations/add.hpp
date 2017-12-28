
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_ADDITION_HPP_
#define _MGCPP_OPERATIONS_ADDITION_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <cstdlib>

namespace mgcpp
{
    namespace strict
    {
        template<typename LhsDenseMat,
                 typename RhsDenseMat,
                 typename Type,
                 size_t DeviceId>
        inline device_matrix<Type, DeviceId,
                             typename LhsDenseMat::allocator_type>
        add(dense_matrix<LhsDenseMat, Type, DeviceId> const& lhs,
            dense_matrix<RhsDenseMat, Type, DeviceId> const& rhs);

        template<typename LhsDenseVec,
                 typename RhsDenseVec,
                 typename Type,
                 size_t Device,
                 alignment Align>
        inline device_vector<Type, Align, Device, 
                             typename LhsDenseVec::allocator_type>
        add(dense_vector<LhsDenseVec, Type, Align, Device> const& first,
            dense_vector<RhsDenseVec, Type, Align, Device> const& second);
    }
}

#include <mgcpp/operations/add.tpp>
#endif
