
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GPU_FORWARD_HPP_
#define _MGCPP_GPU_FORWARD_HPP_

#include <cstdlib>

namespace mgcpp
{
    enum class allignment;
    enum class storage_order { column_major = 0, row_major}; 

    namespace gpu
    {
        template<typename ElemType,
                 size_t DeviceId,
                 storage_order StoreOrder>
        class matrix;

        template<typename ElemType,
                 allignment Allign,
                 size_t DeviceId>
        class vector;
    }
}

#endif
