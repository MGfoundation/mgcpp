#ifndef _MGCPP_EXPRESSIONS_DVEC_REF_EXPR_HPP_
#define _MGCPP_EXPRESSIONS_DVEC_REF_EXPR_HPP_

#include <mgcpp/expressions/dvec_expr.hpp>
#include <mgcpp/vector/forward.hpp>

namespace mgcpp {

template <typename DenseVector, typename Type, size_t DeviceId>
struct dvec_ref_expr : dvec_expr<dvec_ref_expr<DenseVector, Type, DeviceId>> {
  using value_type = Type;
  using result_type = DenseVector;

  DenseVector const& _vec;
  inline dvec_ref_expr(DenseVector const& vec);
  inline DenseVector const& eval() const;
};

template <typename DenseVector, typename Type, size_t DeviceId>
inline dvec_ref_expr<DenseVector, Type, DeviceId> ref(
    dense_vector<DenseVector, Type, DeviceId> const& vec);
}

#endif
