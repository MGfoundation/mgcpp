
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_SYSTEM_CUDA_ERROR_HPP_
#define _MGCPP_SYSTEM_CUDA_ERROR_HPP_

#include <system_error>
#include <type_traits>
#include <string>

#include <cuda_runtime.h>

namespace mgcpp
{
    typedef cudaError_t cuda_error_t;
}

std::error_code
make_error_code(mgcpp::cuda_error_t err) noexcept;

namespace std
{
    template<>
    struct is_error_code_enum<mgcpp::cuda_error_t>
        : public std::true_type {};
}

#endif
