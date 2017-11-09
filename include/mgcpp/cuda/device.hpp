
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUDA_DEVICE_HPP_
#define _MGCPP_CUDA_DEVICE_HPP_

#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

#include <mgcpp/system/error_code.hpp>

#include <cstdlib>

namespace mgcpp
{
    inline outcome::result<void>
    cuda_set_device(size_t device_id) noexcept;
}

#include <mgcpp/cuda/device.ipp>
#endif
