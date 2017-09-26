
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GPU_FORWARD_HPP_
#define _MGCPP_GPU_FORWARD_HPP_


namespace mgcpp
{
    enum class allignment;
    
    namespace gpu
    {
        template<typename ElemType, size_t DeviceId>
        class matrix;

        template<typename ElemType,
                 allignment Allign,
                 size_t DeviceId>
        class vector;
    }
}

#endif
