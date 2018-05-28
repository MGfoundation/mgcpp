
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/bits/hadamard.cuh>
#include <mgcpp/kernels/bits/map.cuh>
#include <mgcpp/kernels/bits/reduce.cuh>
#include <mgcpp/kernels/mgblas_lv1.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/mgblas_error.hpp>

namespace mgcpp {
template <>
inline outcome::result<void> mgblas_vhp(float const* x,
                                        float const* y,
                                        float* z,
                                        size_t n) {
  std::error_code status = mgblas_Svhp(x, y, z, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vhp(double const* x,
                                        double const* y,
                                        double* z,
                                        size_t n) {
  std::error_code status = mgblas_Dvhp(x, y, z, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vab(float* x, size_t n) {
  std::error_code status = mgblas_Svab(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vab(double* x, size_t n) {
  std::error_code status = mgblas_Dvab(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vpr(float const* x, float* y, size_t n) {
  std::error_code status = mgblas_Svpr(x, y, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vpr(double const* x, double* y, size_t n) {
  std::error_code status = mgblas_Dvpr(x, y, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vsin(float* x, size_t n) {
  std::error_code status = mgblas_Svsin(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vsin(double* x, size_t n) {
  std::error_code status = mgblas_Dvsin(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vcos(float* x, size_t n) {
  std::error_code status = mgblas_Svcos(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vcos(double* x, size_t n) {
  std::error_code status = mgblas_Dvcos(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vtan(float* x, size_t n) {
  std::error_code status = mgblas_Svtan(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vtan(double* x, size_t n) {
  std::error_code status = mgblas_Dvtan(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vsinh(float* x, size_t n) {
  std::error_code status = mgblas_Svsinh(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vsinh(double* x, size_t n) {
  std::error_code status = mgblas_Dvsinh(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vcosh(float* x, size_t n) {
  std::error_code status = mgblas_Svcosh(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vcosh(double* x, size_t n) {
  std::error_code status = mgblas_Dvcosh(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vtanh(float* x, size_t n) {
  std::error_code status = mgblas_Svtanh(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vtanh(double* x, size_t n) {
  std::error_code status = mgblas_Dvtanh(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vrelu(float* x, size_t n) {
  std::error_code status = mgblas_Svrelu(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

template <>
inline outcome::result<void> mgblas_vrelu(double* x, size_t n) {
  std::error_code status = mgblas_Dvrelu(x, n);

  if (status != status_t::success)
    return status;
  else
    return outcome::success();
}

}  // namespace mgcpp
