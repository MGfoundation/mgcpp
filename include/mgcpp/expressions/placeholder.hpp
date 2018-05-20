#ifndef PLACEHOLDER_HPP
#define PLACEHOLDER_HPP

#include <mgcpp/expressions/forward.hpp>

namespace mgcpp {

struct placeholder_node_type;

template <size_t PlaceholderID, typename ResultType>
using placeholder_node = generic_expr<placeholder_node_type,
                                      PlaceholderID,
                                      ResultType::template result_expr_type,
                                      ResultType,
                                      0>;
}  // namespace mgcpp

#endif  // PLACEHOLDER_HPP
