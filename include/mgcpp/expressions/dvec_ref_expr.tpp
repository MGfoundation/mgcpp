#include <mgcpp/expressions/dvec_ref_expr.hpp>

namespace mgcpp {

template <typename DenseVector, typename Type, size_t DeviceId>
inline dvec_ref_expr<DenseVector, Type, DeviceId>::dvec_ref_expr(
    DenseVector const& vec)
    : _vec(vec) {}

template <typename DenseVector, typename Type, size_t DeviceId>
inline DenseVector const& dvec_ref_expr<DenseVector, Type, DeviceId>::eval()
    const {
  return _vec;
}

template <typename DenseVector, typename Type, size_t DeviceId>
inline dvec_ref_expr<DenseVector, Type, DeviceId> ref(
    dense_vector<DenseVector, Type, DeviceId> const& vec) {
  return dvec_ref_expr<DenseVector, Type, DeviceId>(~vec);
}
}
