
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_MEAN_HPP_
#define _MGCPP_OPERATIONS_MEAN_HPP_

#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/vector/device_vector.hpp>

namespace mgcpp
{
    namespace strict
    {
        template<typename T,
                 size_t DeviceId,
                 allignment Allign,
                 typename Alloc>
        inline T
        mean(device_vector<T, DeviceId, Allign, Alloc> const& vec);

        template<typename DeviceMatrix,
                 MGCPP_CONCEPT(is_device_matrix<DeviceMatrix>::value)>
        inline typename DeviceMatrix::value_type
        mean(DeviceMatrix const& mat);
    }
}

#include <mgcpp/operations/mean.tpp>
#endif
