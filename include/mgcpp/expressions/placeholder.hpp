#ifndef PLACEHOLDER_HPP
#define PLACEHOLDER_HPP

#include <mgcpp/expressions/forward.hpp>

namespace mgcpp {

struct placeholder_node_type;

template <size_t PlaceholderID,
          template <typename> class ResultExprType,
          typename ResultType>
using placeholder_node =
    generic_expr<placeholder_node_type, PlaceholderID, ResultExprType, ResultType, 0>;
}

#endif  // PLACEHOLDER_HPP
