
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/thread_context.hpp>

namespace mgcpp
{
    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder,
             typename... Args>
    gpu::matrix<ElemType, DeviceId, StoreOrder>
    thread_context::
    make_gpu_matrix(Args... args) const
    {
        return gpu::matrix<ElemType,
                           DeviceId,
                           StoreOrder>(*this, std::forward(args)...);
    }
}
