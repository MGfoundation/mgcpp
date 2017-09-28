
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_THREAD_CONTEXT_HPP_
#define _MGCPP_THREAD_CONTEXT_HPP_

#include <initializer_list>
#include <vector>

#include <mgcpp/gpu/forward.hpp>
#include <mgcpp/context/device_manager.hpp>

namespace mgcpp
{
    class thread_context
    {
    private:
        std::vector<device_manager> _device_managers;

    public:
        thread_context(std::initializer_list<size_t> _devices_used);

        template<typename ElemType, size_t DeviceId, typename... Args>
        inline gpu::matrix<ElemType, DeviceId>
        make_gpu_matrix(Args... args) const;

        cublasHandle_t
        get_cublas() noexcept;
    };
}

#endif
