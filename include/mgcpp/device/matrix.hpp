
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_DEVICE_MATRIX_HPP_
#define _MGCPP_DEVICE_MATRIX_HPP_

#include <mgcpp/allocators/default.hpp>
#include <mgcpp/adapters/adapters.hpp>
#include <mgcpp/context/global_context.hpp>
#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/device/forward.hpp>
#include <mgcpp/global/storage_order.hpp>

#include <cstdlib>
#include <memory>
#include <type_traits>
#include <initializer_list>

namespace mgcpp
{
    template<typename T,
             size_t DeviceId = 0,
             storage_order SO = row_major,
             typename Alloc = mgcpp::default_allocator<T, DeviceId>>
    class device_matrix : public Alloc
    {
    public:
        using this_type = device_matrix<T, DeviceId, SO, Alloc>;
        using result_type = this_type;

    private:
        using Alloc::allocate;
        using Alloc::deallocate;
        using Alloc::device_allocate;
        using Alloc::device_deallocate;
        using Alloc::copy_to_host;
        using Alloc::copy_from_host;

        thread_context* _context;
        std::pair<size_t, size_t> _shape;
        T* _data;
        size_t _capacity;

        inline size_t
        determine_ndim(std::initializer_list<
                       std::initializer_list<T>> const& list) const noexcept;

    public:
        inline
        device_matrix() noexcept;

        inline
        ~device_matrix() noexcept;

        inline
        device_matrix(size_t i, size_t j);

        inline
        device_matrix(size_t i, size_t j, T init);
        
        inline
        device_matrix(size_t i, size_t j, T const* data);

        inline
        device_matrix(
            std::initializer_list<std::initializer_list<T>> const& array);

        template<typename HostMat,
                 typename = typename std::enable_if<adapter<HostMat>::value>::type>
        inline 
        device_matrix(HostMat const& host_mat);

        template<typename HostMat, typename Adapter,
                 typename = typename
                 std::enable_if<std::is_function<Adapter>::value>::type>
        inline 
        device_matrix(HostMat const& host_mat, Adapter& adapter);

        inline
        device_matrix(device_matrix<T, DeviceId, SO, Alloc> const& other);

        inline
        device_matrix(device_matrix<T, DeviceId, SO, Alloc>&& other) noexcept;

        inline device_matrix<T, DeviceId, SO, Alloc>&
        operator=(device_matrix<T, DeviceId, SO, Alloc> const& other);

        inline device_matrix<T, DeviceId, SO, Alloc>&
        operator=(device_matrix<T, DeviceId, SO, Alloc>&& other) noexcept;

        inline device_matrix<T, DeviceId, SO, Alloc>&
        zero();

        inline device_matrix<T, DeviceId, SO, Alloc>&
        resize(size_t i, size_t j);

        inline device_matrix<T, DeviceId, SO, Alloc>&
        resize(size_t i, size_t j, T init);

        inline void
        copy_to_host(T* host_p) const;

        inline T
        check_value(size_t i, size_t j) const;

        inline T const*
        data() const noexcept;

        inline T*
        data_mutable() noexcept;

        inline size_t
        capacity() const noexcept;

        inline thread_context*
        context() const noexcept;

        inline T*
        release_data() noexcept;

        inline std::pair<size_t, size_t> const&
        shape() const noexcept;
    };
}

#include <mgcpp/device/matrix.tpp>
#endif
