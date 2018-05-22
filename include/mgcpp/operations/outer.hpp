#ifndef OUTER_HPP
#define OUTER_HPP

#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

namespace mgcpp {
namespace strict {
template <typename LhsDenseVec,
          typename RhsDenseVec,
          typename Type,
          size_t DeviceId>
inline decltype(auto) outer(
    dense_vector<LhsDenseVec, Type, DeviceId> const& lhs,
    dense_vector<RhsDenseVec, Type, DeviceId> const& rhs);
}
}  // namespace mgcpp

#include <mgcpp/operations/outer.hpp>
#endif  // OUTER_HPP
