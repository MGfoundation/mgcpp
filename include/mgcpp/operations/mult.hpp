
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_MULTIPLICATION_HPP_
#define _MGCPP_OPERATIONS_MULTIPLICATION_HPP_

#include <mgcpp/gpu/matrix.hpp>

namespace mgcpp
{
    template<typename ElemType,
             size_t DeviceId,
             storage_order StoreOrder>
    gpu::matrix
    mult(gpu::matrix<ElemType, DeviceId, StoreOrder> const& first,
         gpu::matrix<ElemType, DeviceId, StoreOrder> const& second);

    template<typename ElemType,
             size_t IdFirst, size_t IdSecond,
             storange_order StoreOrder>
    gpu::matrix
    mult(gpu::matrix<ElemType, IdFirst, StoreOrder> const& first,
         gpu::matrix<ElemType, IdSecond, StoreOrder> const& second);

    template<typename ElemType,
             size_t DeviceId,
             storage_order StorOrder>
    void
    mult_assign(
        gpu::matrix<ElemType, DeviceId, StoreOrder>& first,
        gpu::matrix<ElemType, DeviceId, StoreOrder> const& second);

    template<typename ElemType,
             size_t IdFirst, size_t IdSecond,
             storage_order StoreOrder>
    void
    mult_assign(
        gpu::matrix<ElemType, IdFirst, StoreOrder>& first,
        gpu::matrix<ElemType, IdSecond, StoreOrder> const& second);
}

#include <mgcpp/operations/mult.tpp>
#endif
