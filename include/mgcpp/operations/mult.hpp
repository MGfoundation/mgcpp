
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_MULTIPLICATION_HPP_
#define _MGCPP_OPERATIONS_MULTIPLICATION_HPP_

#include <mgcpp/device/forward.hpp>

namespace mgcpp
{
    namespace strict
    {
        template<typename T, size_t Device, storage_order SO, typename Alloc>
        inline device_matrix<T, Device, SO, Alloc>
        mult(device_matrix<T, Device, SO, Alloc> const& first,
             device_matrix<T, Device, SO, Alloc> const& second);

        template<typename T, size_t Device, allignment Allign, typename Alloc>
        inline device_vector<T, Device, Allign, Alloc>
        mult(device_vector<T, Device, Allign, Alloc> const& first,
             device_vector<T, Device, Allign, Alloc> const& second);

        template<typename T, size_t Device, allignment Allign, typename Alloc>
        inline device_vector<T, Device, Allign, Alloc>
        mult(T scalar,
             device_vector<T, Device, Allign, Alloc> const& vec);


        // template<typename T, size_t Device, storage_order SO>
        // void
        // mult_assign(gpu::matrix<T, Device, SO>& first,
        //             gpu::matrix<T, Device, SO> const& second);
    }
}

#include <mgcpp/operations/mult.tpp>
#endif
