
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_VECTOR_DEVICE_VECTOR_HPP_
#define _MGCPP_VECTOR_DEVICE_VECTOR_HPP_

#include <mgcpp/allocators/default.hpp>
#include <mgcpp/adapters/adapters.hpp>
#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/system/concept.hpp>
#include <mgcpp/device/forward.hpp>
#include <mgcpp/global/allignment.hpp>

#include <cstdlib>
#include <initializer_list>
#include <type_traits>

namespace mgcpp
{
    template<typename Type,
             size_t DeviceId = 0,
             allignment Allign = row,
             typename Alloc = mgcpp::default_allocator<Type, DeviceId>>
    class device_vector
    {
    public:
        using this_type = device_vector<Type, DeviceId, Allign, Alloc>;
        using value_type = Type;
        using pointer = value_type*;
        using result_type = this_type;
        using allocator_type = Alloc;

    private:
        thread_context* _context;
        size_t _shape;
        Alloc _allocator;
        Type* _data;
        size_t _capacity;

    public:
        inline device_vector() noexcept;

        inline ~device_vector() noexcept;

        inline device_vector(Alloc const& alloc) noexcept;

        inline device_vector(size_t size, Alloc const& alloc = Alloc());

        inline device_vector(size_t size, Type init,
                             Alloc const& alloc = Alloc());

        inline device_vector(size_t size, Type const* data,
                             Alloc const& alloc = Alloc());

        inline
        device_vector(std::initializer_list<Type> const& array,
                      Alloc const& alloc = Alloc());

        template<typename HostVec,
                 MGCPP_CONCEPT(adapter<HostVec>::value)>
        inline 
        device_vector(HostVec const& host_mat,
                      Alloc const& alloc = Alloc());

        inline
        device_vector(device_vector<Type, DeviceId, Allign, Alloc> const& other);

        inline
        device_vector(device_vector<Type, DeviceId, Allign, Alloc>&& other) noexcept;

        inline device_vector<Type, DeviceId, Allign, Alloc>&
        operator=(device_vector<Type, DeviceId, Allign, Alloc> const& other);

        inline device_vector<Type, DeviceId, Allign, Alloc>&
        operator=(device_vector<Type, DeviceId, Allign, Alloc>&& other) noexcept;

        inline device_vector<Type, DeviceId, Allign, Alloc>&
        zero();

        inline device_vector<Type, DeviceId, Allign, Alloc>&
        resize(size_t size);

        inline void 
        copy_to_host(Type* host_p) const;

        inline Type
        check_value(size_t i) const;

        inline Type const*
        data() const noexcept;

        inline Type*
        data_mutable() noexcept;

        inline size_t
        capacity() const noexcept;

        inline Type*
        release_data() noexcept;

        inline thread_context*
        context() const noexcept;

        inline size_t
        shape() const noexcept;

        inline size_t
        size() const noexcept;
    };
}

#include <mgcpp/device/vector.tpp>
#endif
