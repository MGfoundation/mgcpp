
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/expressions/mat_expr.hpp>

namespace mgcpp
{
    template<typename DenseMatrix,
             typename Type,
             size_t DeviceId>
    inline decltype(auto)
    eval(dense_matrix<DenseMatrix, Type, DeviceId> const& device_mat)
    {
        return ~device_mat;
    }
}
