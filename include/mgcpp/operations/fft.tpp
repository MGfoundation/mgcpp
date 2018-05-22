
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda_libs/cufft_fft.hpp>
#include <mgcpp/kernels/mgblas_helpers.hpp>
#include <mgcpp/operations/fft.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

namespace mgcpp {
template <typename DeviceVec, typename Type, size_t DeviceId>
decltype(auto) strict::rfft(
    dense_vector<DeviceVec, Type, DeviceId> const& vec) {
  using allocator_type = typename DeviceVec::allocator_type;
  using result_allocator_type =
      typename allocator_type::template rebind_alloc<complex<Type>>;

  auto const& dev_vec = ~vec;

  size_t fft_size = dev_vec.size();
  size_t output_size = fft_size / 2 + 1;

  auto result = device_vector<complex<Type>, DeviceId, result_allocator_type>(
      output_size);

  auto status =
      mgcpp::cufft::rfft(fft_size, dev_vec.data(), result.data_mutable());
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  return result;
}

template <typename DeviceVec, typename Type, size_t DeviceId>
decltype(auto) strict::irfft(
    dense_vector<DeviceVec, complex<Type>, DeviceId> const& vec,
    int n) {
  using allocator_type = typename DeviceVec::allocator_type;
  using result_allocator_type =
      typename allocator_type::template rebind_alloc<Type>;

  auto const& dev_vec = ~vec;

  size_t fft_size = n;
  if (n < 0) {
    fft_size = (dev_vec.size() - 1) * 2;
  } else if (fft_size / 2 + 1 > dev_vec.size()) {
    // Pad vector with zeroes
    auto new_shape = fft_size / 2 + 1;
    auto padded = dev_vec;
    padded.resize(new_shape);
  }
  size_t output_size = fft_size;

  auto result =
      device_vector<Type, DeviceId, result_allocator_type>(output_size);

  auto status =
      mgcpp::cufft::irfft(fft_size, dev_vec.data(), result.data_mutable());
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  // Normalize the result
  result = mgcpp::strict::mult(static_cast<Type>(1. / fft_size), result);
  return result;
}

template <typename DeviceVec, typename Type, size_t DeviceId>
decltype(auto) strict::cfft(
    dense_vector<DeviceVec, complex<Type>, DeviceId> const& vec,
    fft_direction direction) {
  using allocator_type = typename DeviceVec::allocator_type;
  using result_allocator_type =
      typename allocator_type::template rebind_alloc<complex<Type>>;

  auto const& dev_vec = ~vec;

  size_t fft_size = dev_vec.size();
  size_t output_size = fft_size;

  auto result = device_vector<complex<Type>, DeviceId, result_allocator_type>(
      output_size);

  cufft::fft_direction dir;
  if (direction == fft_direction::forward)
    dir = cufft::fft_direction::forward;
  else
    dir = cufft::fft_direction::inverse;

  auto status =
      mgcpp::cufft::cfft(fft_size, dev_vec.data(), result.data_mutable(), dir);
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  // Normalize the result
  if (direction == fft_direction::inverse)
    result = mgcpp::strict::mult(static_cast<Type>(1. / fft_size), result);

  return result;
}

template <typename DeviceMat, typename Type, size_t DeviceId>
decltype(auto) strict::rfft(
    dense_matrix<DeviceMat, Type, DeviceId> const& mat) {
  using allocator_type = typename DeviceMat::allocator_type;
  using result_allocator_type =
      typename allocator_type::template rebind_alloc<complex<Type>>;

  auto const& dev_mat = ~mat;

  auto fft_size = dev_mat.shape();
  auto output_size = make_shape(fft_size[0] / 2 + 1, fft_size[1]);

  auto result = device_matrix<complex<Type>, DeviceId, result_allocator_type>(
      output_size);

  auto status = mgcpp::cufft::rfft2(fft_size[0], fft_size[1], dev_mat.data(),
                                    result.data_mutable());
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  return result;
}

template <typename DeviceMat, typename Type, size_t DeviceId>
decltype(auto) strict::irfft(
    dense_matrix<DeviceMat, complex<Type>, DeviceId> const& mat,
    int n) {
  using allocator_type = typename DeviceMat::allocator_type;
  using result_allocator_type =
      typename allocator_type::template rebind_alloc<Type>;

  auto const& dev_mat = ~mat;

  auto fft_size = make_shape(n, dev_mat.shape()[1]);
  if (n < 0)
    fft_size[0] = (dev_mat.shape()[0] - 1) * 2;
  else if (fft_size[0] / 2 + 1 > dev_mat.shape()[1]) {
    // FIXME: zero-pad input to length floor(n/2)+1
    MGCPP_THROW_RUNTIME_ERROR("Zero-pad FFT unimplemented");
  }
  auto output_size = fft_size;

  auto result =
      device_matrix<Type, DeviceId, result_allocator_type>(output_size);

  auto status = mgcpp::cufft::irfft2(fft_size[0], fft_size[1], dev_mat.data(),
                                     result.data_mutable());
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  // Normalize the result
  result = mgcpp::strict::mult(
      static_cast<Type>(1. / fft_size[0] / fft_size[1]), result);
  return result;
}

template <typename DeviceMat, typename Type, size_t DeviceId>
decltype(auto) strict::cfft(
    dense_matrix<DeviceMat, complex<Type>, DeviceId> const& mat,
    fft_direction direction) {
  using allocator_type = typename DeviceMat::allocator_type;
  using result_allocator_type =
      typename allocator_type::template rebind_alloc<complex<Type>>;

  auto const& dev_mat = ~mat;

  auto fft_size = dev_mat.shape();
  auto output_size = fft_size;

  auto result = device_matrix<complex<Type>, DeviceId, result_allocator_type>(
      output_size);

  cufft::fft_direction dir;
  if (direction == fft_direction::forward)
    dir = cufft::fft_direction::forward;
  else
    dir = cufft::fft_direction::inverse;

  auto status = mgcpp::cufft::cfft2(fft_size[0], fft_size[1], dev_mat.data(),
                                    result.data_mutable(), dir);
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  // Normalize the result
  if (direction == fft_direction::inverse)
    result = mgcpp::strict::mult(
        static_cast<Type>(1. / fft_size[0] / fft_size[1]), result);
  return result;
}
}  // namespace mgcpp
