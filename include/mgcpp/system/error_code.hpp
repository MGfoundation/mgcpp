
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_SYSTEM_ERROR_CODE_
#define _MGCPP_SYSTEM_ERROR_CODE_

#include <mgcpp/system/cuda_error.hpp>
#include <mgcpp/system/cublas_error.hpp>

#include <system_error>
#include <type_traits>
#include <string>

namespace mgcpp
{

    enum class error_t
    {
        success = 0
    };

    namespace internal
    {
        class mgcpp_error_category :public std::error_category
        {
        public:
            const char*
            name() const noexcept override;

            std::string
            message(int ev) const override;

            bool
            equivalent(std::error_code const& err,
                       int condition) const noexcept override;
        };
    }
}

namespace std
{
    template <>
    struct is_error_condition_enum<error_t> : std::true_type {};
}

#endif
