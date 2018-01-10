
#ifndef _MGCPP_GLOBAL_SHAPE_HPP_
#define _MGCPP_GLOBAL_SHAPE_HPP_

#include <array>

namespace mgcpp
{
    template<size_t Dims>
    using shape = std::array<size_t, Dims>;

    template<typename ... Types>
    shape<sizeof...(Types)> make_shape(Types ... args);
}

#include <mgcpp/global/shape.tpp>

#endif
