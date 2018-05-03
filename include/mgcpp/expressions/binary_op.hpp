#ifndef BINARY_OP_HPP
#define BINARY_OP_HPP

#include <utility>
#include <memory>

namespace mgcpp {

template <int OpID, typename LhsExpr, typename RhsExpr, template<typename> class ResultExprType, typename ResultType>
struct binary_op
    : public ResultExprType<binary_op<OpID, LhsExpr, RhsExpr, ResultExprType, ResultType>> {

    using lhs_expr_type = typename std::decay<LhsExpr>::type;
    using rhs_expr_type = typename std::decay<RhsExpr>::type;

    using result_type = ResultType;

    LhsExpr _lhs;
    RhsExpr _rhs;

    inline binary_op(LhsExpr const& lhs, RhsExpr const& rhs) noexcept
        : _lhs(lhs), _rhs(rhs) {}
    inline binary_op(LhsExpr&& lhs, RhsExpr&& rhs) noexcept
        : _lhs(std::move(lhs)), _rhs(std::move(rhs)) {}

    inline result_type eval() const;

protected:
    mutable std::shared_ptr<std::unique_ptr<result_type>> cache_ptr = std::make_shared<std::unique_ptr<result_type>>(nullptr);
};

}

#endif // BINARY_OP_HPP
