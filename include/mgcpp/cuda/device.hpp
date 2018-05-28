
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUDA_DEVICE_HPP_
#define _MGCPP_CUDA_DEVICE_HPP_

#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/outcome.hpp>

#include <cstdlib>

namespace mgcpp {
outcome::result<void> cuda_set_device(size_t device_id) noexcept;
}

#endif
