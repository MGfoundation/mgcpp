
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_SYSTEM_MGCPP_KERNEL_ERROR_HPP_
#define _MGCPP_SYSTEM_MGCPP_KERNEL_ERROR_HPP_

#include <mgcpp/kernels/kernel_status.hpp>

#include <system_error>
#include <type_traits>

namespace mgcpp
{
    std::error_code
    make_error_code(mgcpp::kernel_status_t err) noexcept;
}

namespace std
{
    template<>
    struct is_error_code_enum<mgcpp::kernel_status_t>
        : public std::true_type {};
}

#endif
