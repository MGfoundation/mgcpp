
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/mgblas_lv1.hpp>
#include <mgcpp/operations/elemwise.hpp>
#include <mgcpp/system/exception.hpp>

#include <cstdlib>

namespace mgcpp {

template <typename Type,
          outcome::result<void> (*Function)(Type*, size_t),
          typename DenseVec,
          size_t DeviceId>
inline device_vector<Type, DeviceId, typename DenseVec::allocator_type>
strict::elemwise(dense_vector<DenseVec, Type, DeviceId> const& vec) {
  using allocator_type = typename DenseVec::allocator_type;

  auto const& original_vec = ~vec;

  auto set_device_status = cuda_set_device(DeviceId);
  if (!set_device_status) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_status.error());
  }

  size_t n = original_vec.shape();

  auto result = device_vector<Type, DeviceId, allocator_type>(original_vec);
  auto status = Function(result.data_mutable(), n);
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  return result;
}

template <typename Type,
          outcome::result<void> (*Function)(Type*, size_t),
          typename DenseMat,
          size_t DeviceId>
inline device_matrix<Type, DeviceId, typename DenseMat::allocator_type>
strict::elemwise(dense_matrix<DenseMat, Type, DeviceId> const& mat) {
  using allocator_type = typename DenseMat::allocator_type;

  auto const& original_mat = ~mat;

  auto set_device_status = cuda_set_device(DeviceId);
  if (!set_device_status) {
    MGCPP_THROW_SYSTEM_ERROR(set_device_status.error());
  }

  auto shape = original_mat.shape();

  auto result = device_matrix<Type, DeviceId, allocator_type>(original_mat);
  auto status = Function(result.data_mutable(), shape[0] * shape[1]);
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  return result;
}

template <typename DenseVec, typename Type, size_t DeviceId>
decltype(auto) strict::abs(dense_vector<DenseVec, Type, DeviceId> const& vec) {
  return elemwise<Type, mgblas_vab>(vec);
}

template <typename DenseMat, typename Type, size_t DeviceId>
decltype(auto) strict::abs(dense_matrix<DenseMat, Type, DeviceId> const& mat) {
  return elemwise<Type, mgblas_vab>(mat);
}

template <typename DenseVec, typename Type, size_t DeviceId>
decltype(auto) strict::sin(dense_vector<DenseVec, Type, DeviceId> const& vec) {
  return elemwise<Type, mgblas_vsin>(vec);
}

template <typename DenseMat, typename Type, size_t DeviceId>
decltype(auto) strict::sin(dense_matrix<DenseMat, Type, DeviceId> const& mat) {
  return elemwise<Type, mgblas_vsin>(mat);
}

template <typename DenseVec, typename Type, size_t DeviceId>
decltype(auto) strict::cos(dense_vector<DenseVec, Type, DeviceId> const& vec) {
  return elemwise<Type, mgblas_vcos>(vec);
}

template <typename DenseMat, typename Type, size_t DeviceId>
decltype(auto) strict::cos(dense_matrix<DenseMat, Type, DeviceId> const& mat) {
  return elemwise<Type, mgblas_vcos>(mat);
}

template <typename DenseVec, typename Type, size_t DeviceId>
decltype(auto) strict::tan(dense_vector<DenseVec, Type, DeviceId> const& vec) {
  return elemwise<Type, mgblas_vtan>(vec);
}

template <typename DenseMat, typename Type, size_t DeviceId>
decltype(auto) strict::tan(dense_matrix<DenseMat, Type, DeviceId> const& mat) {
  return elemwise<Type, mgblas_vtan>(mat);
}

template <typename DenseVec, typename Type, size_t DeviceId>
decltype(auto) strict::sinh(dense_vector<DenseVec, Type, DeviceId> const& vec) {
  return elemwise<Type, mgblas_vsinh>(vec);
}

template <typename DenseMat, typename Type, size_t DeviceId>
decltype(auto) strict::sinh(dense_matrix<DenseMat, Type, DeviceId> const& mat) {
  return elemwise<Type, mgblas_vsinh>(mat);
}

template <typename DenseVec, typename Type, size_t DeviceId>
decltype(auto) strict::cosh(dense_vector<DenseVec, Type, DeviceId> const& vec) {
  return elemwise<Type, mgblas_vcosh>(vec);
}

template <typename DenseMat, typename Type, size_t DeviceId>
decltype(auto) strict::cosh(dense_matrix<DenseMat, Type, DeviceId> const& mat) {
  return elemwise<Type, mgblas_vcosh>(mat);
}

template <typename DenseVec, typename Type, size_t DeviceId>
decltype(auto) strict::tanh(dense_vector<DenseVec, Type, DeviceId> const& vec) {
  return elemwise<Type, mgblas_vtanh>(vec);
}

template <typename DenseMat, typename Type, size_t DeviceId>
decltype(auto) strict::tanh(dense_matrix<DenseMat, Type, DeviceId> const& mat) {
  return elemwise<Type, mgblas_vtanh>(mat);
}

template <typename DenseVec, typename Type, size_t DeviceId>
decltype(auto) strict::relu(dense_vector<DenseVec, Type, DeviceId> const& vec) {
  return elemwise<Type, mgblas_vrelu>(vec);
}

template <typename DenseMat, typename Type, size_t DeviceId>
decltype(auto) strict::relu(dense_matrix<DenseMat, Type, DeviceId> const& mat) {
  return elemwise<Type, mgblas_vrelu>(mat);
}
}  // namespace mgcpp
