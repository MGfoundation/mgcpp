
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GPU_MATRIX_HPP_
#define _MGCPP_GPU_MATRIX_HPP_

#include <mgcpp/allocators/default.hpp>
#include <mgcpp/context/global_context.hpp>
#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/device/forward.hpp>
#include <mgcpp/global/shape.hpp>
#include <mgcpp/global/storage_order.hpp>
#include <mgcpp/host/forward.hpp>

namespace mgcpp
{
    template<typename T,
             size_t DeviceId = 0,
             storage_order SO = row_major,
             typename Alloc = mgcpp::default_allocator<T>>
    class device_matrix : public mgcpp::default_allocator<T>
    {
    private:
        thread_context* _context;
        matrix_shape _shape;
        T* _data;

    public:
        inline device_matrix() noexcept;

        inline ~device_matrix() noexcept;

        inline device_matrix(size_t i, size_t j);

        inline device_matrix(size_t i, size_t j, T init);
        
        inline device_matrix(size_t i, size_t j, T const* data);

        inline
        device_matrix(mgcpp::host_matrix<T, SO> const& cpu_mat);

        inline
        device_matrix(mgcpp::device_matrix<T, DeviceId, SO, Alloc> const& other);

        inline
        device_matrix(mgcpp::device_matrix<T, DeviceId, SO, Alloc>&& other) noexcept;

        inline mgcpp::device_matrix<T, DeviceId, SO, Alloc>&
        operator=(mgcpp::device_matrix<T, DeviceId, SO, Alloc> const& other);

        inline mgcpp::device_matrix<T, DeviceId, SO, Alloc>&
        operator=(mgcpp::device_matrix<T, DeviceId, SO, Alloc>&& other) noexcept;

        inline mgcpp::device_matrix<T, DeviceId, SO, Alloc>&
        zero();

        inline mgcpp::device_matrix<T, DeviceId, SO, Alloc>&
        resize(size_t i, size_t j);

        inline mgcpp::device_matrix<T, DeviceId, SO, Alloc>&
        resize(size_t i, size_t j, T init);

        inline mgcpp::device_matrix<T, DeviceId, SO, Alloc>&
        operator=(mgcpp::host_matrix<T, SO> const& cpu_mat);

        inline mgcpp::host_matrix<T, SO>  
        copy_to_host() const;

        inline T
        check_value(size_t i, size_t j) const;

        inline T const*
        data() const noexcept;

        inline T*
        data_mutable() noexcept;

        inline mgcpp::thread_context*
        context() const noexcept;

        inline T*
        release_data() noexcept;

        inline mgcpp::matrix_shape const&
        shape() const noexcept;
    };
}

#include <mgcpp/device/matrix.tpp>
#endif
