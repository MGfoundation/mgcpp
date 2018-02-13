
#include <mgcpp/global/shape.hpp>

#include <algorithm>

namespace mgcpp
{
    template<size_t Dims>
    shape<Dims>::shape ()
        : dims{}
    {}

    template<size_t Dims>
    shape<Dims>::shape (std::initializer_list<size_t> list)
    {
        std::copy(list.begin(), list.end(), dims);
    }

    template<size_t Dims>
    size_t shape<Dims>::operator[] (size_t idx) const
    {
        return dims[idx];
    }

    template<size_t Dims>
    size_t &shape<Dims>::operator[] (size_t idx)
    {
        return dims[idx];
    }

    template<size_t Dims>
    bool shape<Dims>::operator== (shape const& rhs) const
    {
        for (auto i = 0u; i < Dims; ++i)
            if (dims[i] != rhs[i]) return false;
        return true;
    }

    template<size_t Dims>
    template <std::size_t N>
    size_t shape<Dims>::get() const
    {
        return dims[N];
    }

    template<typename ... Types>
    shape<sizeof...(Types)> make_shape(Types ... args)
    {
        return {static_cast<size_t>(args)...};
    }
}
