
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/operations/mult.hpp>

namespace mgcpp
{
    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix
    mult(gpu::matrix<ElemType, DeviceId, StoreOrder> const& first,
         gpu::matrix<ElemType, DeviceId, StoreOrder> const& second);
}
