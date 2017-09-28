
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/system/error_code.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace mgcpp
{
    const char*
    internal::mgcpp_error_category::
    name() const noexcept
    {
        return "mgcpp";
    }

    std::string
    internal::mgcpp_error_category::
    message(int ev) const
    {
        switch(static_cast<mgcpp::error_t>(ev))
        {
        case error_t::success:
            return "successfully operated";
            break;
        }
        return "";
    }

    bool
    internal::mgcpp_error_category::
    equivalent(std::error_code const& err,
               int condition) const noexcept
    {
        switch(static_cast<mgcpp::error_t>(condition))
        {
        case error_t::success:
            if(err == cublas_error_t::CUBLAS_STATUS_SUCCESS ||
               err == cuda_error_t::cudaSuccess)
                return true;
            else
                return false;
            break; 
        }
        return false;
    }
}
