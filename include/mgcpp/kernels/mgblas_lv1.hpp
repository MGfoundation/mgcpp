
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_KERNELS_MGBLAS_LV1_HPP_
#define _MGCPP_KERNELS_MGBLAS_LV1_HPP_

#include <mgcpp/system/outcome.hpp>

#include <cstdlib>

namespace mgcpp {
/**
 * Hadamard product
 * \param x x
 * \param y y
 * \param z pointer to the result
 * \param n number of elements
 */
template <typename T>
inline outcome::result<void> mgblas_vhp(T const* x, T const* y, T* z, size_t n);

/**
 * Absolute value
 * \param x x
 * \param n number of elements
 */
template <typename T>
inline outcome::result<void> mgblas_vab(T* x, size_t n);

/**
 * Calculates the sum of all elements
 * \param x x
 * \param y result
 * \param n number of elements
 */
template <typename T>
inline outcome::result<void> mgblas_vpr(T const* x, T* y, size_t n);

template <typename T>
inline outcome::result<void> mgblas_vsin(T* x, size_t n);

template <typename T>
inline outcome::result<void> mgblas_vcos(T* x, size_t n);

template <typename T>
inline outcome::result<void> mgblas_vtan(T* x, size_t n);

template <typename T>
inline outcome::result<void> mgblas_vsinh(T* x, size_t n);

template <typename T>
inline outcome::result<void> mgblas_vcosh(T* x, size_t n);

template <typename T>
inline outcome::result<void> mgblas_vtanh(T* x, size_t n);

template <typename T>
inline outcome::result<void> mgblas_vrelu(T* x, size_t n);
}  // namespace mgcpp

#include <mgcpp/kernels/mgblas_lv1.tpp>
#endif
