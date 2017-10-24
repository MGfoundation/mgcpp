
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CPU_FORWARD_HPP_
#define _MGCPP_CPU_FORWARD_HPP_

#include <mgcpp/global/storage_order.hpp>
#include <mgcpp/global/allignment.hpp>

namespace mgcpp
{
    namespace cpu
    {
        template<typename T,
                 storage_order SO = row_major>
        class matrix;

        template<typename T,
                 allignment Allign = row>
        class vector;
    }
}

#endif
