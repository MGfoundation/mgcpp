#include <mgcpp/operations/outer.hpp>

#include <mgcpp/cuda_libs/cublas.hpp>
#include <mgcpp/matrix/device_matrix.hpp>

namespace mgcpp {

template <typename LhsDenseVec, typename RhsDenseVec, typename Type>
inline decltype(auto) strict::outer(
    dense_vector<LhsDenseVec, Type> const& lhs,
    dense_vector<RhsDenseVec, Type> const& rhs) {
  using allocator_type = typename LhsDenseVec::allocator_type;

  auto* context = (~lhs).context();
  auto handle = context->get_cublas_context((~lhs).device_id());

  auto shape = mgcpp::make_shape((~lhs).size(), (~rhs).size());

  device_matrix<Type, allocator_type> result(shape);

  const Type alpha = static_cast<Type>(1);

  auto status =
      cublas::ger(handle, shape[0], shape[1], &alpha, (~lhs).data(), 1,
                  (~rhs).data(), 1, result.data_mutable(), shape[0]);
  if (!status) {
    MGCPP_THROW_SYSTEM_ERROR(status.error());
  }

  return result;
}
}  // namespace mgcpp
