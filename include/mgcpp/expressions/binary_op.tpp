#ifndef BINARY_OP_TPP
#define BINARY_OP_TPP

#include <mgcpp/expressions/binary_op.hpp>
#include <mgcpp/expressions/evaluator.hpp>

namespace mgcpp {

template <int OpID, typename LhsExpr, typename RhsExpr, template<typename> class ResultExprType, typename ResultType>
typename binary_op<OpID, LhsExpr, RhsExpr, ResultExprType, ResultType>::result_type
binary_op<OpID, LhsExpr, RhsExpr, ResultExprType, ResultType>::eval() const
{
    if (!_cache)
        _cache = std::make_shared<result_type>(evaluator::eval(*this));
    return *_cache;
}

}

#endif // BINARY_OP_TPP
