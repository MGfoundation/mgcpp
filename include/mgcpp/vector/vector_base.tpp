
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/vector/vector_base.hpp>

namespace mgcpp
{
    template<typename VectorType,
             typename Type,
             size_t DeviceId,
             allignment Allign>
    VectorType const&
    vector_base<VectorType, Type, DeviceId, Allign>::
    operator~() const noexcept
    { return *static_cast<VectorType const*>(this); };

    template<typename VectorType,
             typename Type,
             size_t DeviceId,
             allignment Allign>
    VectorType&
    vector_base<VectorType, Type, DeviceId, Allign>::
    operator~() noexcept
    { return *static_cast<VectorType*>(this); };
}
