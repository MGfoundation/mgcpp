
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda/device.hpp>

#include <cuda_runtime.h>

namespace mgcpp {
outcome::result<void> cuda_set_device(size_t device_id) noexcept {
  std::error_code err_code = cudaSetDevice(static_cast<int>(device_id));

  if (err_code != status_t::success)
    return err_code;

  return outcome::success();
}
}  // namespace mgcpp
