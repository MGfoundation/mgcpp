
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cuda_libs/cublas.hpp>
#include <mgcpp/matrix/device_matrix.hpp>
#include <mgcpp/operations/trans.hpp>

namespace mgcpp {

template <typename DenseMat, typename Type>
decltype(auto) strict::trans(
    dense_matrix<DenseMat, Type> const& mat) {
  using allocator_type = typename DenseMat::allocator_type;

  auto const& dmat = ~mat;
  auto* context = dmat.context();
  auto handle = context->get_cublas_context(dmat.allocator()._device_id);

  auto shape = dmat.shape();

  auto m = shape[0];
  auto n = shape[1];

  Type alpha = 1;
  Type beta = 0;

  // switch places
  auto result = device_matrix<Type, allocator_type>({n, m});

  Type* null = nullptr;
  auto status =
      cublas::geam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, dmat.data(),
                   m, &beta, null, n, result.data_mutable(), n);

  if (!status)
    MGCPP_THROW_SYSTEM_ERROR(status.error());

  return result;
}
}  // namespace mgcpp
