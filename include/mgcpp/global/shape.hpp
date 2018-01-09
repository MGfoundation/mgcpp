
#ifndef _MGCPP_GLOBAL_SHAPE_HPP_
#define _MGCPP_GLOBAL_SHAPE_HPP_

#include <array>

namespace mgcpp
{
    template<size_t Dims>
    struct shape
    {
        size_t dims[Dims];
        size_t operator[] (size_t i) const;
        size_t &operator[] (size_t i);
        bool operator==(shape<Dims> const &rhs) const;
    };
}

#include <mgcpp/global/shape.tpp>

#endif
