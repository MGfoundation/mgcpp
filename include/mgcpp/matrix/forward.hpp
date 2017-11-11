
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_MATRIX_FORWARD_HPP_
#define _MGCPP_MATRIX_FORWARD_HPP_

namespace mgcpp
{
    template <typename MatrixType,
              typename Type,
              size_t DeviceId>
    class matrix;

    template <typename DenseMatrixType,
              typename Type,
              size_t DeviceId>
    class dense_matrix;

    template <typename Type,
              size_t DeviceId,
              typename Alloc>
    class device_matrix;
}

#endif
