
#include <mgcpp/global/shape.hpp>

#include <algorithm>

namespace mgcpp
{
    template<typename ... Types>
    shape<sizeof...(Types)> make_shape(Types ... args)
    {
        return {static_cast<size_t>(args)...};
    }
}
