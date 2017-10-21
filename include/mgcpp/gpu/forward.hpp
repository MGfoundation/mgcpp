
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GPU_FORWARD_HPP_
#define _MGCPP_GPU_FORWARD_HPP_

#include <cstdlib>
#include <mgcpp/global/storage_order.hpp>
#include <mgcpp/global/allignment.hpp>

namespace mgcpp
{
    namespace gpu
    {
        template<typename ElemType,
                 size_t DeviceId = 0,
                 storage_order StoreOrder = row_major>
        class matrix;

        template<typename ElemType,
                 allignment Allign,
                 size_t DeviceId>
        class vector;
    }
}

#endif
