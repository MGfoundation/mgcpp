
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
template <typename ElemType, typename>
outcome::result<ElemType*> cuda_malloc(size_t size) noexcept {
  void* ptr = nullptr;
  std::error_code err_code = cudaMalloc(&ptr, size * sizeof(ElemType));

  if (err_code != status_t::success)
    return err_code;

  return static_cast<ElemType*>(ptr);
}
#define INSTANTIATE_CUDA_MALLOC(type) \
  template outcome::result<type*> cuda_malloc<type>(size_t) noexcept;
INSTANTIATE_CUDA_MALLOC(float)
INSTANTIATE_CUDA_MALLOC(double)
INSTANTIATE_CUDA_MALLOC(::half)
INSTANTIATE_CUDA_MALLOC(::cuComplex)
INSTANTIATE_CUDA_MALLOC(::cuDoubleComplex)
#undef INSTANTIATE_CUDA_MALLOC

template <typename ElemType>
outcome::result<void> cuda_free(ElemType* ptr) noexcept {
  std::error_code err_code = cudaFree(ptr);

  if (err_code != status_t::success)
    return err_code;

  return outcome::success();
}
#define INSTANTIATE_CUDA_FREE(type) \
  template outcome::result<void> cuda_free<type>(type*) noexcept;
INSTANTIATE_CUDA_FREE(float)
INSTANTIATE_CUDA_FREE(double)
INSTANTIATE_CUDA_FREE(::half)
INSTANTIATE_CUDA_FREE(::cuComplex)
INSTANTIATE_CUDA_FREE(::cuDoubleComplex)
#undef INSTANTIATE_CUDA_FREE

template <typename ElemType>
outcome::result<ElemType*> malloc_pinned(size_t count) noexcept {
  void* ptr = nullptr;
  std::error_code err_code = cudaMallocHost(&ptr, count * sizeof(ElemType));

  if (err_code != status_t::success)
    return err_code;

  return static_cast<ElemType*>(ptr);
}
#define INSTANTIATE_MALLOC_PINNED(type) \
  template outcome::result<type*> malloc_pinned<type>(size_t) noexcept;
INSTANTIATE_MALLOC_PINNED(float)
INSTANTIATE_MALLOC_PINNED(double)
INSTANTIATE_MALLOC_PINNED(::half)
INSTANTIATE_MALLOC_PINNED(::cuComplex)
INSTANTIATE_MALLOC_PINNED(::cuDoubleComplex)
#undef INSTANTIATE_MALLOC_PINNED

template <typename ElemType>
outcome::result<void> free_pinned(ElemType* ptr) noexcept {
  std::error_code err_code = cudaFreeHost(ptr);

  if (err_code != status_t::success)
    return err_code;

  return outcome::success();
}
#define INSTANTIATE_FREE_PINNED(type) \
  template outcome::result<void> free_pinned<type>(type*) noexcept;
INSTANTIATE_FREE_PINNED(float)
INSTANTIATE_FREE_PINNED(double)
INSTANTIATE_FREE_PINNED(::half)
INSTANTIATE_FREE_PINNED(::cuComplex)
INSTANTIATE_FREE_PINNED(::cuDoubleComplex)
#undef INSTANTIATE_FREE_PINNED

template <typename ElemType>
outcome::result<void> cuda_memset_to_zero(ElemType* ptr,
                                          size_t count) noexcept {
  std::error_code err_code =
      cudaMemset(static_cast<void*>(ptr), 0, sizeof(ElemType) * count);

  if (err_code != status_t::success)
    return err_code;

  return outcome::success();
}
#define INSTANTIATE_CUDA_MEMSET_TO_ZERO(type)                     \
  template outcome::result<void> cuda_memset_to_zero<type>(type*, \
                                                           size_t) noexcept;
INSTANTIATE_CUDA_MEMSET_TO_ZERO(float)
INSTANTIATE_CUDA_MEMSET_TO_ZERO(double)
INSTANTIATE_CUDA_MEMSET_TO_ZERO(::half)
INSTANTIATE_CUDA_MEMSET_TO_ZERO(::cuComplex)
INSTANTIATE_CUDA_MEMSET_TO_ZERO(::cuDoubleComplex)
#undef INSTANTIATE_CUDA_MEMSET_TO_ZERO

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
#define INSTANTIATE_CUDA_MEMCPY(type)                                          \
  template outcome::result<void> cuda_memcpy<type>(type*, type const*, size_t, \
                                                   cuda_memcpy_kind) noexcept;
INSTANTIATE_CUDA_MEMCPY(float)
INSTANTIATE_CUDA_MEMCPY(double)
INSTANTIATE_CUDA_MEMCPY(::half)
INSTANTIATE_CUDA_MEMCPY(::cuComplex)
INSTANTIATE_CUDA_MEMCPY(::cuDoubleComplex)
#undef INSTANTIATE_CUDA_MEMCPY

// Specializations of cuda_memcpy
outcome::result<void> cuda_memcpy(cuComplex* to,
                                  std::complex<float> const* from,
                                  size_t count,
                                  cuda_memcpy_kind kind) noexcept {
  return cuda_memcpy(to, reinterpret_cast<cuComplex const*>(from), count, kind);
}

outcome::result<void> cuda_memcpy(std::complex<float>* to,
                                  cuComplex const* from,
                                  size_t count,
                                  cuda_memcpy_kind kind) noexcept {
  return cuda_memcpy(reinterpret_cast<cuComplex*>(to), from, count, kind);
}

outcome::result<void> cuda_memcpy(cuDoubleComplex* to,
                                  std::complex<double> const* from,
                                  size_t count,
                                  cuda_memcpy_kind kind) noexcept {
  return cuda_memcpy(to, reinterpret_cast<cuDoubleComplex const*>(from), count,
                     kind);
}

outcome::result<void> cuda_memcpy(std::complex<double>* to,
                                  cuDoubleComplex const* from,
                                  size_t count,
                                  cuda_memcpy_kind kind) noexcept {
  return cuda_memcpy(reinterpret_cast<cuDoubleComplex*>(to), from, count, kind);
}

#ifdef USE_HALF
outcome::result<void> cuda_memcpy(__half* to,
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

outcome::result<void> cuda_memcpy(float* to,
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

outcome::result<std::pair<free_mem_t, total_mem_t>>
cuda_mem_get_info() noexcept {
  size_t free_memory = 0;
  size_t total_memory = 0;

  std::error_code status = cudaMemGetInfo(&free_memory, &total_memory);

  if (status != status_t::success)
    return status;
  else
    return std::make_pair(free_memory, total_memory);
}
#endif
}  // namespace mgcpp
