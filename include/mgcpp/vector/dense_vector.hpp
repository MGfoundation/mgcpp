
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_VECTOR_DENSE_VECTOR_HPP_
#define _MGCPP_VECTOR_DENSE_VECTOR_HPP_

#include <mgcpp/vector/vector_base.hpp>

namespace mgcpp
{
    template<typename DenseVecType,
             typename Type,
             alignment Align,
             size_t DeviceId>
    class dense_vector :
        public vector_base<DenseVecType, Type, Align, DeviceId>
    {

    };
}

#endif
