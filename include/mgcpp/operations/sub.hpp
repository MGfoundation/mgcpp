
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_SUBSTRACTION_HPP_
#define _MGCPP_OPERATIONS_SUBSTRACTION_HPP_

#include <mgcpp/device/forward.hpp>

namespace mgcpp
{
    namespace strict
    {
        template<typename T, size_t Device, allignment Allign, typename Alloc>
        inline device_vector<T, Device, Allign, Alloc>
        sub(device_vector<T, Device, Allign, Alloc> const& first,
            device_vector<T, Device, Allign, Alloc> const& second);

        template<typename LhsMat, typename RhsMat,
                 MGCPP_CONCEPT(is_device_matrix<LhsMat>::value
                               && is_device_matrix<RhsMat>::value)>
        inline device_matrix<typename LhsMat::value_type,
                             LhsMat::device_id,
                             typename LhsMat::allocator_type>
        sub(LhsMat const& first, RhsMat const& second);
    }
}

#include <mgcpp/operations/sub.tpp>
#endif
