
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CPU_FORWARD_HPP_
#define _MGCPP_CPU_FORWARD_HPP_

namespace mgcpp
{
    enum class storage_order;

    namespace cpu
    {
        template<typename ElemType,
                 storage_order StoreOrder>
        class matrix;
    }
}

#endif
