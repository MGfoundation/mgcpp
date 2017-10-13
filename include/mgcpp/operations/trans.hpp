
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_TRANS_HPP_
#define _MGCPP_OPERATIONS_TRANS_HPP_

#include <mgcpp/gpu/forward.hpp>

namespace mgcpp
{
    namespace strict
    {
        template<typename T,
                 size_t Device,
                 storage_order SO>
        gpu::matrix<T, Device, SO>
        trans(gpu::matrix<T, Device, SO> const& mat);
    }
}

#include <mgcpp/operations/trans.tpp>
#endif
