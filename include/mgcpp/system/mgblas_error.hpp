
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_SYSTEM_MGCPP_KERNEL_ERROR_HPP_
#define _MGCPP_SYSTEM_MGCPP_KERNEL_ERROR_HPP_

#include <mgcpp/kernels/mgblas_error_code.hpp>

#include <system_error>
#include <type_traits>

namespace mgcpp
{
    std::error_code
    make_error_code(mgcpp::mgblas_error_t err) noexcept;
}

namespace std
{
    template<>
    struct is_error_code_enum<mgcpp::mgblas_error_t>
        : public std::true_type {};
}

#endif
