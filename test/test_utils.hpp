
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TEST_TEST_UTILS_HPP_
#define _MGCPP_TEST_TEST_UTILS_HPP_

#include <gtest/gtest.h>
#include <complex>
#include <cstdlib>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/vector/device_vector.hpp>
#include <random>
#include <type_traits>
#include <vector>

#include "cpu_matrix.hpp"

#define MGCPP_RAND_SEED 1

namespace internal {
double uniform_rand();
}

template <typename Type>
struct complex_value_type;

template <typename Type>
struct complex_value_type<std::complex<Type>> {
  using type = Type;
};

template <typename Iter, typename Type>
void random_fill_impl(Iter begin, Iter end) {
  for (Iter it = begin; it != end; ++it)
    *it = internal::uniform_rand();
}

template <typename Iter,
          typename Type,
          typename = typename complex_value_type<Type>::type>
void random_fill_impl(Iter begin, Iter end) {
  using value_type = typename complex_value_type<Type>::type;
  for (Iter it = begin; it != end; ++it)
    *it = std::complex<value_type>(internal::uniform_rand(),
                                   internal::uniform_rand());
}

template <typename Iter>
void random_fill(Iter begin, Iter end) {
  using value_type = decltype(*begin);
  random_fill_impl<Iter, value_type>(begin, end);
}

template <typename Type>
void random_matrix(Type& mat) {
  using value_type = typename Type::value_type;
  auto shape = mat.shape();
  auto size = shape[0] * shape[1];
  cpu_matrix<value_type> buf(shape[0], shape[1]);
  auto* ptr = buf.data();
  random_fill(ptr, ptr + size);
  mat = Type(buf);
}

template <typename Type>
void random_vector(Type& vec) {
  using value_type = typename Type::value_type;
  auto size = vec.size();
  cpu_vector<value_type> buf(size);
  auto* ptr = buf.data();
  random_fill(ptr, ptr + size);
  vec = Type(buf);
}

#endif
