
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_DEVICE_MATRIX_HPP_
#define _MGCPP_DEVICE_MATRIX_HPP_

#include <mgcpp/adapters/adapters.hpp>
#include <mgcpp/allocators/default.hpp>
#include <mgcpp/context/global_context.hpp>
#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/device/forward.hpp>
#include <mgcpp/global/storage_order.hpp>
#include <mgcpp/system/concept.hpp>
#include <mgcpp/type_traits/device_matrix.hpp>

#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <type_traits>

namespace mgcpp
{
    template<typename Type,
             size_t DeviceId = 0,
             typename Alloc = mgcpp::default_allocator<Type, DeviceId>>
    class device_matrix : public Alloc
    {
    public:
        using this_type = device_matrix<Type, DeviceId, Alloc>;
        using value_type = Type;
        using pointer = value_type*;
        using result_type = this_type;
        using allocator_type = Alloc;

        static size_t const device_id = DeviceId;

    private:
        using Alloc::allocate;
        using Alloc::deallocate;
        using Alloc::device_allocate;
        using Alloc::device_deallocate;
        using Alloc::copy_to_host;
        using Alloc::copy_from_host;

        thread_context* _context;
        std::pair<size_t, size_t> _shape;
        Type* _data;
        size_t _capacity;

        inline size_t
        determine_ndim(std::initializer_list<
                       std::initializer_list<Type>> const& list) const noexcept;

    public:
        inline
        device_matrix() noexcept;

        inline
        ~device_matrix() noexcept;

        inline
        device_matrix(size_t i, size_t j);

        inline
        device_matrix(size_t i, size_t j, Type init);
        
        inline
        device_matrix(size_t i, size_t j, Type const* data);

        inline
        device_matrix(
            std::initializer_list<std::initializer_list<Type>> const& array);

        // inline
        // device_matrix(std::initializer_list<device_vector<Type>> const& array);

        template<typename HostMat,
                 MGCPP_CONCEPT(adapter<HostMat>::value)>
        inline 
        device_matrix(HostMat const& host_mat);

        inline
        device_matrix(device_matrix<Type, DeviceId, Alloc> const& other);

        inline
        device_matrix(device_matrix<Type, DeviceId, Alloc>&& other) noexcept;

        inline device_matrix<Type, DeviceId, Alloc>&
        operator=(device_matrix<Type, DeviceId, Alloc> const& other);

        inline device_matrix<Type, DeviceId, Alloc>&
        operator=(device_matrix<Type, DeviceId, Alloc>&& other) noexcept;

        inline device_matrix<Type, DeviceId, Alloc>&
        zero();

        inline device_matrix<Type, DeviceId, Alloc>&
        resize(size_t i, size_t j);

        inline device_matrix<Type, DeviceId, Alloc>&
        resize(size_t i, size_t j, Type init);

        inline void
        copy_to_host(Type* host_p) const;

        inline Type
        check_value(size_t i, size_t j) const;

        inline Type const*
        data() const noexcept;

        inline Type*
        data_mutable() noexcept;

        inline size_t
        capacity() const noexcept;

        inline thread_context*
        context() const noexcept;

        inline Type*
        release_data() noexcept;

        inline std::pair<size_t, size_t> const&
        shape() const noexcept;
    };
}

#include <mgcpp/device/matrix.tpp>
#endif
