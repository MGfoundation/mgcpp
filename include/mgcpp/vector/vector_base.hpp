
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_VECTOR_VECTOR_BASE_HPP_
#define _MGCPP_VECTOR_VECTOR_BASE_HPP_

#include <cstdlib>

namespace mgcpp
{
    enum alignment : bool
    {
        row = true,
        column = false
    };

    template<typename VectorType,
             typename Type,
             alignment Align,
             size_t DeviceId>
    struct vector_base
    {
        inline VectorType const&
        operator~() const noexcept;

        inline VectorType&
        operator~() noexcept;
    };
}

#include <mgcpp/vector/vector_base.tpp>
#endif
