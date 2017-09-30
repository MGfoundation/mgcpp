
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef MGCPP_GPU_GPU_PTR_HPP
#define MGCPP_GPU_GPU_PTR_HPP

namespace mgcpp
{
    template <typename ElemType>
    struct gpu_ptr
    {
        gpu_ptr(gpu_ptr const& other) = delete;
        gpu_ptr(gpu_ptr&& other) = delete;
        gpu_ptr& operator=(gpu_ptr&& other) = delete;
        gpu_ptr& operator=(gpu_ptr const& other) = delete;

        gpu_ptr(ElemType* data_) noexcept;

        ElemType* data;
    }
}
