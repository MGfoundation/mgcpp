
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cpu/vector.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    template<typename T,
             allignment Allign>
    cpu::vector<T, Allign>::
    vector() noexcept
    : _data(nullptr),
        _size(0),
        _released(true) {}
    
    template<typename T,
             allignment Allign>
    cpu::vector<T, Allign>::
    vector(size_t size) 
        : _data(nullptr),
          _size(size),
          _released(true)
    {
        T* ptr = (T*)malloc(sizeof(T) * _size);
        if(!ptr)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        _released = false;
        
        _data = ptr;
    }

    template<typename T,
             allignment Allign>
    cpu::vector<T, Allign>::
    vector(size_t size, T init)
        : _data(nullptr),
          _size(size),
          _released(true)
    {
        T* ptr = (T*)malloc(sizeof(T) * _size);
        if(!ptr)
        {
            MGCPP_THROW_BAD_ALLOC;
        }
        _released = false;

        std::fill(ptr, ptr + _size, init);
        
        _data = ptr;
    }

    template<typename T,
             allignment Allign>
    cpu::vector<T, Allign>::
    vector(size_t size, T* data) noexcept
        : _data(data),
          _size(size), 
          _released(false) {}

    template<typename T,
             allignment Allign>
    cpu::vector<T, Allign>::
    vector(cpu::vector<T, Allign> const& other)
        : _data(nullptr),
          _size(0), 
          _released(true) 
    {

    }

    template<typename T,
             allignment Allign>
    cpu::vector<T, Allign>::
    vector(std::initializer_list<T> const& array) noexcept
    {
        
    }

    template<typename T,
             allignment Allign>
    cpu::vector<T, Allign>::
    vector(cpu::vector<T, Allign>&& other) noexcept
        : _data(nullptr),
          _size(0), 
          _released(true) 
    {
            
    }

    template<typename T,
             allignment Allign>
    cpu::vector<T, Allign>&
    cpu::vector<T, Allign>::
    operator=(cpu::vector<T, Allign> const& other)
    {
        
    }

    template<typename T,
             allignment Allign>
    cpu::vector<T, Allign>&
    cpu::vector<T, Allign>::
    operator=(cpu::vector<T, Allign>&& other) noexcept
    {
        
    }

    template<typename T,
             allignment Allign>
    T
    cpu::vector<T, Allign>::
    at(size_t i) const
    {
        if(i >= _size)
        {
            MGCPP_THROW_OUT_OF_RANGE("index out of range");
        }

        return _data[i];
    }

    template<typename T,
             allignment Allign>
    T&
    cpu::vector<T, Allign>::
    at(size_t i)
    {
        if(i >= _size)
        {
            MGCPP_THROW_OUT_OF_RANGE("index out of range");
        }

        return _data[i];
    }

    template<typename T,
             allignment Allign>
    T
    cpu::vector<T, Allign>::
    operator[](size_t i) const noexcept
    {
        return _data[i];
    }

    template<typename T,
             allignment Allign>
    T&
    cpu::vector<T, Allign>::
    operator[](size_t i) noexcept
    {
        return _data[i];
    }

    template<typename T,
             allignment Allign>
    T const*
    cpu::vector<T, Allign>::
    data() const
    {
        return _data; 
    }

    template<typename T,
             allignment Allign>
    T*
    cpu::vector<T, Allign>::
    data_mutable() noexcept
    {
        return _data;
    }

    template<typename T,
             allignment Allign>
    size_t
    cpu::vector<T, Allign>::
    shape() const noexcept
    {
        return _size; 
    }

    template<typename T,
             allignment Allign>
    size_t
    cpu::vector<T, Allign>::
    size() const noexcept
    {
        return _size; 
    }

    template<typename T,
             allignment Allign>
    T*
    cpu::vector<T, Allign>::
    released_data()
    {
        _released= true;
        return _data;
    }


    template<typename T,
             allignment Allign>
    cpu::vector<T, Allign>::
    ~vector() noexcept
    {
        if(!_released)
            (void)free(_data);
    }
}
