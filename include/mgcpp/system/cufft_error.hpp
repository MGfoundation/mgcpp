
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_SYSTEM_CUFFT_ERROR_HPP_
#define _MGCPP_SYSTEM_CUFFT_ERROR_HPP_

#include <cufft.h>

#include <system_error>
#include <type_traits>
#include <string>

namespace mgcpp
{
    using cufft_error_t = cufftResult;
}

std::error_code
make_error_code(mgcpp::cufft_error_t err) noexcept;

namespace std
{
    template<>
    struct is_error_code_enum<mgcpp::cufft_error_t>
        : public std::true_type {};
}

#endif

