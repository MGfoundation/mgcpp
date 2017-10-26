
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_EXPRESSIONS_GPU_MAT_HPP_
#define _MGCPP_EXPRESSIONS_GPU_MAT_HPP_

#include <type_traits>

#include <mgcpp/device/forward.hpp>

namespace mgcpp
{
    template<typename T>
    struct is_gpu_matrix : std::false_type {};

    template<typename T, size_t DeviceId, storage_order SO>
    struct is_gpu_matrix<mgcpp::device_matrix<T, DeviceId, SO>>
        : std::true_type {};

    template<typename Mat>
    struct assert_gpu_matrix
    {
        using result =
            typename std::enable_if<
            is_gpu_matrix<
                typename std::decay<Mat>::type>::value>::type;
    };
}

#endif
