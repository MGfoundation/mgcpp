
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/dvec_expr.hpp>

namespace mgcpp
{
    template<typename DenseMatrix,
             typename Type,
             alignment Align,
             size_t DeviceId>
    inline decltype(auto)
    eval(dense_vector<DenseMatrix, Type, Align, DeviceId> const& device_mat)
    {
        return ~device_mat;
    }
}
