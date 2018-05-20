
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CUDA_MEMORY_HPP_
#define _MGCPP_CUDA_MEMORY_HPP_

#include <mgcpp/system/outcome.hpp>
#include <mgcpp/system/error_code.hpp>

#include <complex>
#include <cstdlib>
#include <new>
#include <type_traits>

namespace mgcpp {

template <typename ElemType,
          typename = std::enable_if<std::is_arithmetic<ElemType>::value>>
outcome::result<ElemType*> cuda_malloc(size_t size) noexcept;

template <typename ElemType>
outcome::result<void> cuda_free(ElemType* ptr) noexcept;

template <typename ElemType>
outcome::result<void> cuda_memset(ElemType* ptr,
                                  ElemType value,
                                  size_t count) noexcept;

template <typename ElemType>
outcome::result<ElemType*> malloc_pinned(size_t count) noexcept;

template <typename ElemType>
outcome::result<void> free_pinned(ElemType* ptr) noexcept;

enum class cuda_memcpy_kind;

template <typename ElemType>
outcome::result<void> cuda_memcpy(ElemType* to,
                                  ElemType const* from,
                                  size_t count,
                                  cuda_memcpy_kind kind) noexcept;

inline outcome::result<void> cuda_memcpy(cuComplex* to,
                                         std::complex<float> const* from,
                                         size_t count,
                                         cuda_memcpy_kind kind) noexcept;

inline outcome::result<void> cuda_memcpy(std::complex<float>* to,
                                         cuComplex const* from,
                                         size_t count,
                                         cuda_memcpy_kind kind) noexcept;

inline outcome::result<void> cuda_memcpy(cuDoubleComplex* to,
                                         std::complex<double> const* from,
                                         size_t count,
                                         cuda_memcpy_kind kind) noexcept;

inline outcome::result<void> cuda_memcpy(std::complex<double>* to,
                                         cuDoubleComplex const* from,
                                         size_t count,
                                         cuda_memcpy_kind kind) noexcept;

inline outcome::result<void> cuda_memcpy(__half* to,
                                         float const* from,
                                         size_t count,
                                         cuda_memcpy_kind kind) noexcept;

inline outcome::result<void> cuda_memcpy(float* to,
                                         __half const* from,
                                         size_t count,
                                         cuda_memcpy_kind kind) noexcept;

using free_mem_t = size_t;
using total_mem_t = size_t;

outcome::result<
    std::pair<free_mem_t, total_mem_t>> inline cuda_mem_get_info() noexcept;
}  // namespace mgcpp

#include <mgcpp/cuda/memory.tpp>
#endif
