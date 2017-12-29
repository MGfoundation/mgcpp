
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_MATRIX_DEVICE_MATRIX_HPP_
#define _MGCPP_MATRIX_DEVICE_MATRIX_HPP_

#include <mgcpp/adapters/adapters.hpp>
#include <mgcpp/allocators/default.hpp>
#include <mgcpp/context/global_context.hpp>
#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/matrix/column_view.hpp>
#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/matrix/row_view.hpp>
#include <mgcpp/system/concept.hpp>
#include <mgcpp/type_traits/device_pointer_type.hpp>
#include <mgcpp/type_traits/host_value_type.hpp>

#include <cstdlib>
#include <initializer_list>
#include <type_traits>

namespace mgcpp
{
    template<typename Type,
             size_t DeviceId = 0,
             typename Alloc = mgcpp::default_allocator<Type, DeviceId>>
    class device_matrix
        : public dense_matrix<device_matrix<Type, DeviceId, Alloc>,
                              Type,
                              DeviceId>
    {
    public:
        using this_type = device_matrix<Type, DeviceId, Alloc>;
        using value_type = typename value_type<Type>::type;
        using pointer = Type*;
        using const_pointer = Type const*;
        using device_pointer = typename device_pointer<Type>::type;
        using const_device_pointer = typename const_device_pointer<Type>::type;
        using result_type = this_type;
        using allocator_type = Alloc;
        size_t const device_id = DeviceId;

    private:
        thread_context* _context;
        std::pair<size_t, size_t> _shape;
        Alloc _allocator;
        pointer _data;
        size_t _capacity;

        inline size_t
        determine_ndim(std::initializer_list<
                       std::initializer_list<value_type>> const& list) const noexcept;

    public:
        inline device_matrix() noexcept;

        inline
        ~device_matrix() noexcept;

        inline
        device_matrix(Alloc const& alloc);

        inline
        device_matrix(size_t i, size_t j,
                      Alloc const& alloc = Alloc());

        inline
        device_matrix(size_t i, size_t j, value_type init,
                      Alloc const& alloc = Alloc());
        
        inline
        device_matrix(size_t i, size_t j, const_pointer data,
                      Alloc const& alloc = Alloc());

        inline
        device_matrix(std::initializer_list<
                      std::initializer_list<value_type>> const& array,
                      Alloc const& alloc = Alloc());

        template<typename HostMat,
                 MGCPP_CONCEPT(adapter<HostMat>::value)>
        inline 
        device_matrix(HostMat const& host_mat,
                      Alloc const& alloc = Alloc());

        inline
        device_matrix(device_matrix<Type, DeviceId, Alloc> const& other);

        template<typename DenseMatrix>
        inline
        device_matrix(dense_matrix<DenseMatrix, Type, DeviceId> const& other);

        inline
        device_matrix(device_matrix<Type, DeviceId, Alloc>&& other) noexcept;

        inline device_matrix<Type, DeviceId, Alloc>&
        operator=(device_matrix<Type, DeviceId, Alloc> const& other);

        template<typename DenseMatrix>
        inline device_matrix<Type, DeviceId, Alloc>&
        operator=(dense_matrix<DenseMatrix, Type, DeviceId> const& other);

        inline device_matrix<Type, DeviceId, Alloc>&
        operator=(device_matrix<Type, DeviceId, Alloc>&& other) noexcept;

        inline device_matrix<Type, DeviceId, Alloc>&
        zero();

        inline device_matrix<Type, DeviceId, Alloc>&
        resize(size_t i, size_t j);

        inline device_matrix<Type, DeviceId, Alloc>&
        resize(size_t i, size_t j, value_type init);

        inline column_view<this_type, Type, DeviceId>
        column(size_t i) noexcept;

        inline row_view<this_type, Type, DeviceId>
        row(size_t i) noexcept;

        inline void
        copy_to_host(pointer host_p) const;

        inline value_type
        check_value(size_t i, size_t j) const;

        inline void
        set_value(size_t i, size_t j, value_type value);

        inline const_device_pointer
        data() const noexcept;

        inline device_pointer
        data_mutable() noexcept;

        inline size_t
        capacity() const noexcept;

        inline thread_context*
        context() const noexcept;

        inline device_pointer
        release_data() noexcept;

        inline Alloc&
        allocator() noexcept;

        inline std::pair<size_t, size_t> const&
        shape() const noexcept;
    };
}

#include <mgcpp/matrix/device_matrix.tpp>
#endif
