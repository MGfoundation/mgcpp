
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/system/cuda_error.hpp>

#include <cuda_runtime.h>

namespace mgcpp {
class cuda_error_category_t : public std::error_category {
 public:
  const char* name() const noexcept override;

  std::string message(int ev) const override;
} cuda_error_category;

const char* cuda_error_category_t::name() const noexcept {
  return "cuda";
}

std::string cuda_error_category_t::message(int ev) const {
  return "internal cuda error: " +
         std::string(cudaGetErrorString(static_cast<cuda_error_t>(ev)));
}
}  // namespace mgcpp

std::error_code make_error_code(mgcpp::cuda_error_t err) noexcept {
  return {static_cast<int>(err), mgcpp::cuda_error_category};
}
