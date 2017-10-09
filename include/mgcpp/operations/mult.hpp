
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_MULTIPLICATION_HPP_
#define _MGCPP_OPERATIONS_MULTIPLICATION_HPP_

#define private public
#include <mgcpp/gpu/matrix.hpp>
#include <mgcpp/global/storage_order.hpp>

namespace mgcpp
{
    template<typename T,
             size_t Device,
             storage_order SO>
    gpu::matrix<T, Device, SO>
    mult(gpu::matrix<T, Device, SO> const& first,
         gpu::matrix<T, Device, SO> const& second);

    // template<typename T,
    //          size_t Device>
    // gpu::matrix
    // mult(gpu::matrix<T, Device, storage_order::column_major> const& first,
    //      gpu::matrix<T, Device, storage_order::column_major> const& second);

    // template<typename T,
    //          size_t IdFirst, size_t IdSecond,
    //          storange_order StoreOrder>
    // gpu::matrix
    // mult(gpu::matrix<T, IdFirst, StoreOrder> const& first,
    //      gpu::matrix<T, IdSecond, StoreOrder> const& second);

    template<typename T,
             size_t Device,
             storage_order SO>
    void
    mult_assign(gpu::matrix<T, Device, SO>& first,
                gpu::matrix<T, Device, SO> const& second);

    // template<typename T,
    //          size_t IdFirst, size_t IdSecond,
    //          storage_order StoreOrder>
    // void
    // mult_assign(
    //     gpu::matrix<T, IdFirst, StoreOrder>& first,
    //     gpu::matrix<T, IdSecond, StoreOrder> const& second);
}

#include <mgcpp/operations/mult.tpp>
#endif
