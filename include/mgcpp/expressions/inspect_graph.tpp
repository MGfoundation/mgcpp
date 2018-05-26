#include <mgcpp/expressions/inspect_graph.hpp>

#include <mgcpp/expressions/constant_expr.hpp>
#include <mgcpp/expressions/dmat_dmat_mult.hpp>
#include <mgcpp/expressions/dmat_reduce_expr.hpp>
#include <mgcpp/expressions/dmat_trans_expr.hpp>
#include <mgcpp/expressions/placeholder.hpp>
#include <mgcpp/expressions/symbolic_shape_expr.hpp>

namespace mgcpp {

namespace internal {
template <size_t PlaceholderID, typename ResultType>
std::ostream& inspect(std::ostream& os,
                      placeholder_node<PlaceholderID, ResultType>) {
  os << "Ph" << PlaceholderID;
  return os;
}

template <typename LhsExpr, typename RhsExpr>
std::ostream& inspect(std::ostream& os,
                      dmat_dmat_mult_expr<LhsExpr, RhsExpr> const& expr) {
  os << (~expr).first() << " * " << (~expr).second();
  return os;
}

template <typename LhsExpr, typename RhsExpr>
std::ostream& inspect(std::ostream& os,
                      dmat_dmat_add_expr<LhsExpr, RhsExpr> const& expr) {
  os << (~expr).first() << " + " << (~expr).second();
  return os;
}

template <typename LhsExpr, typename RhsExpr>
std::ostream& inspect(std::ostream& os,
                      dmat_dvec_mult_expr<LhsExpr, RhsExpr> const& expr) {
  os << (~expr).first() << " * " << (~expr).second();
  return os;
}

template <typename LhsExpr, typename RhsExpr>
std::ostream& inspect(std::ostream& os,
                      dvec_dvec_add_expr<LhsExpr, RhsExpr> const& expr) {
  os << (~expr).first() << " + " << (~expr).second();
  return os;
}

template <typename LhsExpr, typename RhsExpr>
std::ostream& inspect(std::ostream& os,
                      dvec_dvec_outer_expr<LhsExpr, RhsExpr> const& expr) {
  os << (~expr).first() << " (*) " << (~expr).second();
  return os;
}

template <typename Expr>
std::ostream& inspect(std::ostream& os, zeros_mat_expr<Expr> const& expr) {
  os << "ZerosLike{" << (~expr).first() << "}";
  return os;
}

template <typename Expr>
std::ostream& inspect(std::ostream& os, zeros_vec_expr<Expr> const& expr) {
  os << "ZerosLike{" << (~expr).first() << "}";
  return os;
}

template <typename Expr>
std::ostream& inspect(std::ostream& os, ones_mat_expr<Expr> const& expr) {
  os << "OnesLike{" << (~expr).first() << "}";
  return os;
}

template <typename Expr>
std::ostream& inspect(std::ostream& os, ones_vec_expr<Expr> const& expr) {
  os << "OnesLike{" << (~expr).first() << "}";
  return os;
}

template <typename Expr>
std::ostream& inspect(std::ostream& os, symbolic_shape_expr<Expr> const& expr) {
  os << "Shape{" << (~expr).first() << "}";
  return os;
}

template <typename Expr>
std::ostream& inspect(std::ostream& os,
                      dmat_reduce_sum_expr<Expr> const& expr) {
  os << "MatReduceSum{" << (~expr).first() << "}";
  return os;
}

template <typename Expr>
std::ostream& inspect(std::ostream& os,
                      dvec_reduce_sum_expr<Expr> const& expr) {
  os << "VecReduceSum{" << (~expr).first() << "}";
  return os;
}

template <typename Expr>
std::ostream& inspect(std::ostream& os, dmat_trans_expr<Expr> const& expr) {
  os << "Transpose{" << (~expr).first() << "}";
  return os;
}

}  // namespace internal

template <typename Expr>
std::ostream& operator<<(std::ostream& os, expression<Expr> const& expr) {
  os << "(";
  internal::inspect(os, ~expr);
  os << ")";
  return os;
}

}  // namespace mgcpp
