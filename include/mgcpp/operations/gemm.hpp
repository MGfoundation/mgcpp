
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_OPERATIONS_GEMM_HPP_
#define _MGCPP_OPERATIONS_GEMM_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/type_traits/type_traits.hpp>

#include <cstdlib>

namespace mgcpp
{
    namespace strict
    {
        template<typename ADense,
                 typename BDense,
                 typename CDense,
                 typename Type,
                 size_t DeviceId>
        inline decltype(auto)
        gemm(dense_matrix<ADense, Type, DeviceId> const& A,
             dense_matrix<BDense, Type, DeviceId> const& B,
             dense_matrix<CDense, Type, DeviceId> const& C);

        template<typename ADense,
                 typename BDense,
                 typename CDense,
                 typename Type,
                 size_t DeviceId,
                 typename ScalarType,
                 typename = typename
                 std::enable_if<is_scalar<ScalarType>::value>::type>
        inline decltype(auto)
        gemm(ScalarType alpha,
             dense_matrix<ADense, Type, DeviceId> const& A,
             dense_matrix<BDense, Type, DeviceId> const& B,
             ScalarType beta,
             dense_matrix<CDense, Type, DeviceId> const& C);
    }
}

#include <mgcpp/operations/gemm.tpp>
#endif
