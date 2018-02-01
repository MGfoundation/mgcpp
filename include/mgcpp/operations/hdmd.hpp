
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_HADAMARD_HPP_
#define _MGCPP_OPERATIONS_HADAMARD_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <cstdlib>

namespace mgcpp
{
    namespace strict
    {
        template<typename LhsDenseVec,
                 typename RhsDenseVec,
                 typename Type,
                 alignment Align,
                 size_t DeviceId>
        inline decltype(auto)
        hdmd(dense_vector<LhsDenseVec, Type, Align, DeviceId> const& lhs,
             dense_vector<RhsDenseVec, Type, Align, DeviceId> const& rhs);
    }
}

#include <mgcpp/operations/hdmd.tpp>
#endif
