
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_KERNELS_MGBLAS_ERROR_HPP_
#define _MGCPP_KERNELS_MGBLAS_ERROR_HPP_

namespace mgcpp {
enum mgblas_error_t {
  success = 0,
  index_out_of_range = 1,
  invalid_range = 2,
  memory_allocation_failure = 3,
  device_to_host_memcpy_failure = 4
};
}

#endif
