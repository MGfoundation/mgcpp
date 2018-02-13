
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_SUBSTRACTION_HPP_
#define _MGCPP_OPERATIONS_SUBSTRACTION_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <cstdlib>

namespace mgcpp
{
    namespace strict
    {
        /**
         * Vector Subtraction.
         * \param lhs the left-hand side
         * \param rhs the right-hand side
         * \returns lhs - rhs
         */
        template<typename LhsDenseVec,
                 typename RhsDenseVec,
                 typename Type,
                 size_t DeviceId>
        inline decltype(auto)
        sub(dense_vector<LhsDenseVec, Type, DeviceId> const& lhs,
            dense_vector<RhsDenseVec, Type, DeviceId> const& rhs);

        /**
         * Matrix Subtraction.
         * \param lhs the left-hand side
         * \param rhs the right-hand side
         * \returns lhs - rhs
         */
        template<typename LhsDenseMat,
                 typename RhsDenseMat,
                 typename Type,
                 size_t DeviceId>
        inline decltype(auto)
        sub(dense_matrix<LhsDenseMat, Type, DeviceId> const& lhs,
            dense_matrix<RhsDenseMat, Type, DeviceId> const& rhs);
    }
}

#include <mgcpp/operations/sub.tpp>
#endif
