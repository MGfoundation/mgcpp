
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_ABSOLUTE_HPP_
#define _MGCPP_OPERATIONS_ABSOLUTE_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/vector/device_vector.hpp>

namespace mgcpp
{
    namespace strict
    {
        template<typename T,
                 size_t Device,
                 allignment Allign,
                 typename Alloc>
        inline device_vector<T, Device, Allign, Alloc>
        abs(device_vector<T, Device, Allign, Alloc> const& vec);

        template<typename DenseMat,
                 typename Type,
                 size_t DeviceId>
        inline device_matrix<Type, DeviceId, typename DenseMat::allocator_type>
        abs(dense_matrix<DenseMat, Type, DeviceId> const& mat);
    }
}

#include <mgcpp/operations/abs.tpp>
#endif
