
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CONTEXT_THREAD_CONTEXT_HPP_
#define _MGCPP_CONTEXT_THREAD_CONTEXT_HPP_

#include <cublas_v2.h>

#include <functional>
#include <memory>
#include <unordered_map>

namespace mgcpp {
template <typename T, typename U>
using hash_table = std::unordered_map<T, U>;

class thread_context {
 private:
  using cublas_handle_unique_ptr =
      std::unique_ptr<cublasContext, std::function<void(cublasHandle_t)>>;

  hash_table<size_t, cublas_handle_unique_ptr> _cublas_handle;

 public:
  thread_context() = default;
  thread_context(thread_context const& other) = delete;
  thread_context& operator=(thread_context const& other) = delete;

  thread_context(thread_context&& other) noexcept;

  thread_context& operator=(thread_context&& other) noexcept;

  cublasHandle_t get_cublas_context(size_t device_id);
};
}  // namespace mgcpp

#endif
