
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUDA_ERROR_HPP_
#define _MGCPP_CUDA_ERROR_HPP_

#include <system_error>
#include <type_traits>
#include <string>

#include <cuda_runtime.h>

namespace mgcpp
{
    typedef cudaError_t cuda_error_t;

    namespace internal
    {
        class cuda_error_category_t : public std::error_category
        {
        public:
            const char*
            name() const noexcept override;

            std::string
            message(int ev) const override;
        };

        extern cuda_error_category_t cuda_error_category;
    }
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
