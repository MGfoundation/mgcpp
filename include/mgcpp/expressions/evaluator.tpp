#ifndef EVALUATOR_TPP
#define EVALUATOR_TPP

#include <mgcpp/expressions/evaluator.hpp>

#include <mgcpp/expressions/dmat_dmat_add.hpp>
#include <mgcpp/expressions/dmat_dmat_mult.hpp>
#include <mgcpp/expressions/dmat_dvec_mult.hpp>

#include <mgcpp/operations/add.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/operations/gemm.hpp>

namespace mgcpp {
namespace internal {

template <typename LhsExpr, typename RhsExpr>
auto eval(mat_mat_add_op<LhsExpr, RhsExpr> const& expr) {
    auto lhs = mgcpp::eval(expr._lhs);
    auto rhs = mgcpp::eval(expr._rhs);

    return strict::add(lhs, rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto eval(mat_mat_mult_op<LhsExpr, RhsExpr> const& expr) {
    auto const& lhs = mgcpp::eval(expr._lhs);
    auto const& rhs = mgcpp::eval(expr._rhs);

    return strict::mult(lhs, rhs);
}

template <typename LhsExpr, typename RhsExpr>
auto eval(mat_vec_mult_op<LhsExpr, RhsExpr> const& expr) {
    auto const& lhs = mgcpp::eval(expr._lhs);
    auto const& rhs = mgcpp::eval(expr._rhs);

    return strict::mult(lhs, rhs);
}
}

template <typename Op>
auto evaluator::eval(Op const& op) {
    return internal::eval(op);
}
}

#endif // EVALUATOR_TPP
