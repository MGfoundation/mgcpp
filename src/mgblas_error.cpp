
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/system/mgblas_error.hpp>
#include <string>

namespace mgcpp
{
    class mgblas_error_category_t
        : public std::error_category
    {
    public:
        const char*
        name() const noexcept override;

        std::string
        message(int ev) const override;
    } mgblas_error_category;


    const char*
    mgblas_error_category_t::
    name() const noexcept
    {
        return "mgblas";
    }

    std::string
    mgblas_error_category_t::
    message(int ev) const
    {
        switch(static_cast<mgblas_error_t>(ev))
        {
        case success:
            return "Operation was executed without problem";
            break;

        case index_out_of_range:
            return "requested operation range is incorrect";
            break;

        case invalid_range:
            return "operation range is invalid";
            break;
        }
        return "";
    }

    std::error_code
    make_error_code(mgcpp::mgblas_error_t err) noexcept
    {
        return {static_cast<int>(err),
                mgcpp::mgblas_error_category};
    }
}
