
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_MAT_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_MAT_EXPR_HPP_

#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/system/concept.hpp>
#include <mgcpp/system/concept.hpp>

#include <cstdlib>
#include <utility>

namespace mgcpp
{
    template<typename T>
    struct mat_expr {};

    template<typename DenseMatrix,
             typename Type,
             size_t DeviceId>
    inline decltype(auto)
    eval(dense_matrix<DenseMatrix, Type, DeviceId> const& device_mat);
}

#include <mgcpp/expressions/mat_expr.tpp>
#endif
