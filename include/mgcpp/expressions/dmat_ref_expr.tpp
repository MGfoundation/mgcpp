#include <mgcpp/expressions/dmat_ref_expr.hpp>

namespace mgcpp
{

template <typename DenseMatrix, typename Type, size_t DeviceId>
inline dmat_ref_expr<DenseMatrix, Type, DeviceId>::dmat_ref_expr(
    DenseMatrix const& mat)
    : _mat(mat) {}

template <typename DenseMatrix, typename Type, size_t DeviceId>
inline DenseMatrix const& dmat_ref_expr<DenseMatrix, Type, DeviceId>::eval()
    const {
  return _mat;
}

template <typename DenseMatrix, typename Type, size_t DeviceId>
inline decltype(auto) eval(
    dmat_ref_expr<DenseMatrix, Type, DeviceId> const& mat) {
  return mat.eval();
}

template <typename DenseMatrix, typename Type, size_t DeviceId>
inline dmat_ref_expr<DenseMatrix, Type, DeviceId> ref(
    dense_matrix<DenseMatrix, Type, DeviceId> const& mat) {
  return dmat_ref_expr<DenseMatrix, Type, DeviceId>(~mat);
}

}
