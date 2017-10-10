
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>

#include <mgcpp/gpu/forward.hpp>

namespace mgcpp
{
    template<typename Lhs, typename Rhs>
    struct is_same_gpu_matrix : std::false_type {};

    template<typename T, size_t Id, storage_order SO>
    struct is_same_gpu_matrix<gpu::matrix<T, Id, SO>,
                              gpu::matrix<T, Id, SO>>
        : std::true_type {};

    template<typename Lhs, typename Rhs>
    struct assert_same_gpu_matrix
    {
        using result = typename std::enable_if<
            is_same_gpu_matrix<Lhs, Rhs>::value>::type;
    };
}
