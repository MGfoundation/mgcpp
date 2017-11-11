
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_SUBSTRACTION_HPP_
#define _MGCPP_OPERATIONS_SUBSTRACTION_HPP_

#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <cstdlib>

namespace mgcpp
{
    namespace strict
    {
        template<typename T, size_t Device, allignment Allign, typename Alloc>
        inline device_vector<T, Device, Allign, Alloc>
        sub(device_vector<T, Device, Allign, Alloc> const& first,
            device_vector<T, Device, Allign, Alloc> const& second);

        template<typename LhsDenseMat,
                 typename RhsDenseMat,
                 typename Type,
                 size_t DeviceId>
        inline device_matrix<Type, DeviceId, typename LhsDenseMat::allocator_type>
        sub(dense_matrix<LhsDenseMat, Type, DeviceId> const& first,
            dense_matrix<RhsDenseMat, Type, DeviceId> const& second);
    }
}

#include <mgcpp/operations/sub.tpp>
#endif
