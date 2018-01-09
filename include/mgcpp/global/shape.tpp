
#include <mgcpp/global/shape.hpp>

namespace mgcpp
{
    template<size_t Dims>
    size_t
    shape<Dims>::
    operator[] (size_t i) const
    {
        return dims[i];
    }

    template<size_t Dims>
    size_t&
    shape<Dims>::
    operator[] (size_t i)
    {
        return dims[i];
    }

    template<size_t Dims>
    bool
    shape<Dims>::
    operator==(shape<Dims> const &rhs) const
    {
        for (auto i = 0u; i < Dims; ++i)
            if (dims[i] != rhs[i]) return false;
        return true;
    }

    template<typename ... Types>
    shape<sizeof...(Types)> make_shape(Types ... args)
    {
        return {static_cast<size_t>(args)...};
    }
}
