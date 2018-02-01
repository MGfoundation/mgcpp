
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/system/error_code.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace mgcpp
{
    class mgcpp_error_category_t
        :public std::error_category
    {
    public:
        const char*
        name() const noexcept override;

        std::string
        message(int ev) const override;

        bool
        equivalent(std::error_code const& err,
                   int condition) const noexcept override;
    } mgcpp_error_category;


    const char*
    mgcpp_error_category_t::
    name() const noexcept
    {
        return "mgcpp";
    }

    std::string
    mgcpp_error_category_t::
    message(int ev) const
    {
        switch(static_cast<mgcpp::status_t>(ev))
        {
        case status_t::success:
            return "successfully operated";
            break;
        }
        return "";
    }

    bool
    mgcpp_error_category_t::
    equivalent(std::error_code const& err,
               int condition) const noexcept
    {
        switch(static_cast<mgcpp::status_t>(condition))
        {
        case status_t::success:
            if(err == cublas_error_t::CUBLAS_STATUS_SUCCESS ||
               err == cuda_error_t::cudaSuccess ||
               err == cufft_error_t::CUFFT_SUCCESS ||
               err == mgblas_error_t::success)
                return true;
            else
                return false;
            break; 
        }
        return false;
    }

    std::error_condition
    make_error_condition(mgcpp::status_t err) noexcept
    {
        return {static_cast<int>(err), mgcpp::mgcpp_error_category};
    }
}

