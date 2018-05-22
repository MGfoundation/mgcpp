#include <mgcpp/operations/outer.hpp>

#include <mgcpp/cuda_libs/cublas.hpp>
#include <mgcpp/matrix/device_matrix.hpp>

namespace mgcpp {

template <typename LhsDenseVec,
          typename RhsDenseVec,
          typename Type,
          size_t DeviceId>
inline decltype(auto) outer(
    dense_vector<LhsDenseVec, Type, DeviceId> const& lhs,
    dense_vector<RhsDenseVec, Type, DeviceId> const& rhs) {
  using allocator_type = typename LhsDenseVec::allocator_type;
  using device_pointer = typename LhsDenseVec::device_pointer;

  auto* context = (~lhs).context();
  auto handle = context->get_cublas_context(DeviceId);

  auto shape = mgcpp::make_shape((~lhs).size(), (~rhs).size());

  device_matrix<Type, DeviceId, allocator_type> result(shape);

  Type alpha = static_cast<Type>(1);

  auto status = cublas::ger(handle, shape[0], shape[1], alpha, (~lhs).data(), 1,
                            (~rhs).data(), 1, result.data_mutable(), shape[0]);
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  return result;
}
}  // namespace mgcpp
