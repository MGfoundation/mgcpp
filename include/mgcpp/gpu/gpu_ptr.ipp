
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/gpu/gpu_ptr.hpp>

namespace mgcpp
{
    template <typename ElemType>
    gpu_ptr::
    gpu_ptr(ElemType* data_) noexcept
        : data(data_) {}

    template <typename ElemType>
    void
    gpu_ptr::
    swap(gpu_ptr* other) noexcept
    {
        ElemType* temp = data;
        data = other.data;
        other.data = temp;
    }
}
