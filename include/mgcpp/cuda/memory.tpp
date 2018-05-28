
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cuda_runtime.h>

#include <mgcpp/cuda/memory.hpp>
#include <mgcpp/kernels/mgblas_helpers.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp {
template <typename ElemType>
outcome::result<ElemType*> cuda_malloc(size_t size) noexcept {
  void* ptr = nullptr;
  std::error_code err_code = cudaMalloc(&ptr, size * sizeof(ElemType));

  if (err_code != status_t::success)
    return err_code;

  return static_cast<ElemType*>(ptr);
}

template <typename ElemType>
outcome::result<void> cuda_free(ElemType* ptr) noexcept {
  std::error_code err_code = cudaFree(ptr);

  if (err_code != status_t::success)
    return err_code;

  return outcome::success();
}

template <typename ElemType>
outcome::result<ElemType*> malloc_pinned(size_t count) noexcept {
  void* ptr = nullptr;
  std::error_code err_code = cudaMallocHost(&ptr, count * sizeof(ElemType));

  if (err_code != status_t::success)
    return err_code;

  return static_cast<ElemType*>(ptr);
}

template <typename ElemType>
outcome::result<void> free_pinned(ElemType* ptr) noexcept {
  std::error_code err_code = cudaFreeHost(ptr);

  if (err_code != status_t::success)
    return err_code;

  return outcome::success();
}

template <typename ElemType>
outcome::result<void> cuda_memset_to_zero(ElemType* ptr,
                                          size_t count) noexcept {
  std::error_code err_code =
      cudaMemset(static_cast<void*>(ptr), 0, sizeof(ElemType) * count);

  if (err_code != status_t::success)
    return err_code;

  return outcome::success();
}

template <typename ElemType>
outcome::result<void> cuda_memcpy(ElemType* to,
                                  ElemType const* from,
                                  size_t count,
                                  cuda_memcpy_kind kind) noexcept {
  std::error_code status =
      cudaMemcpy(static_cast<void*>(to), static_cast<void const*>(from),
                 count * sizeof(ElemType), static_cast<cudaMemcpyKind>(kind));

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

// Specializations of cuda_memcpy
inline outcome::result<void> cuda_memcpy(cuComplex* to,
                                         std::complex<float> const* from,
                                         size_t count,
                                         cuda_memcpy_kind kind) noexcept {
  return cuda_memcpy(to, reinterpret_cast<cuComplex const*>(from), count, kind);
}

inline outcome::result<void> cuda_memcpy(std::complex<float>* to,
                                         cuComplex const* from,
                                         size_t count,
                                         cuda_memcpy_kind kind) noexcept {
  return cuda_memcpy(reinterpret_cast<cuComplex*>(to), from, count, kind);
}

inline outcome::result<void> cuda_memcpy(cuDoubleComplex* to,
                                         std::complex<double> const* from,
                                         size_t count,
                                         cuda_memcpy_kind kind) noexcept {
  return cuda_memcpy(to, reinterpret_cast<cuDoubleComplex const*>(from), count,
                     kind);
}

inline outcome::result<void> cuda_memcpy(std::complex<double>* to,
                                         cuDoubleComplex const* from,
                                         size_t count,
                                         cuda_memcpy_kind kind) noexcept {
  return cuda_memcpy(reinterpret_cast<cuDoubleComplex*>(to), from, count, kind);
}

#ifdef USE_HALF
inline outcome::result<void> cuda_memcpy(__half* to,
                                         float const* from,
                                         size_t count,
                                         cuda_memcpy_kind kind) noexcept {
  auto ptr = cuda_malloc<float>(count);
  if (!ptr) {
    return ptr.error();
  }

  auto cpy_stat = cuda_memcpy(ptr.value(), from, count, kind);
  if (!cpy_stat) {
    return ptr.error();
  }

  // convert float -> half
  auto conv_stat = mgblas_convert_copy(ptr.value(), to, count);
  if (!conv_stat) {
    return conv_stat.error();
  }

  return cuda_free(ptr.value());
}

inline outcome::result<void> cuda_memcpy(float* to,
                                         __half const* from,
                                         size_t count,
                                         cuda_memcpy_kind kind) noexcept {
  auto ptr = cuda_malloc<float>(count);
  if (!ptr) {
    return ptr.error();
  }

  // convert half -> float
  auto conv_stat = mgblas_convert_copy(from, ptr.value(), count);
  if (!conv_stat) {
    return conv_stat.error();
  }

  auto cpy_stat = cuda_memcpy(to, ptr.value(), count, kind);
  if (!cpy_stat) {
    return cpy_stat.error();
  }

  return cuda_free(ptr.value());
}
#endif

inline outcome::result<std::pair<free_mem_t, total_mem_t>>
cuda_mem_get_info() noexcept {
  size_t free_memory = 0;
  size_t total_memory = 0;

  std::error_code status = cudaMemGetInfo(&free_memory, &total_memory);

  if (status != status_t::success)
    return status;
  else
    return std::make_pair(free_memory, total_memory);
}
}  // namespace mgcpp
