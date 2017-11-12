
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_MATRIX_COLUMN_VIEW_HPP_
#define _MGCPP_MATRIX_COLUMN_VIEW_HPP_

#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/type_traits/get_member_trait.hpp>
#include <mgcpp/vector/dense_vector.hpp>

#include <cstdlib>
#include <initializer_list>

namespace mgcpp
{
    template<typename DenseMat,
             typename Type,
             size_t DeviceId>
    class column_view
        : public dense_vector<column_view<DenseMat, Type, DeviceId>,
                              Type,
                              column,
                              DeviceId>
    {
    public:
        using this_type = column_view<DenseMat, Type, DeviceId>;
        using value_type = Type;
        using result_type = this_type;
        using allocator_type = get_allocator<DenseMat>;

    private:
        DenseMat* _matrix;
        size_t i;

    public:
        inline column_view() = delete;

        inline ~column_view() = default;

        inline
        column_view(DenseMat& mat, size_t i);

        inline
        column_view(column_view<DenseMat, Type, DeviceId> const& mat);

        inline
        column_view(column_view<DenseMat, Type, DeviceId>&& mat) noexcept;

        inline column_view<DenseMat, Type, DeviceId>&
        operator=(column_view<DenseMat, Type, DeviceId> const& mat);

        inline column_view<DenseMat, Type, DeviceId>&
        operator=(column_view<DenseMat, Type, DeviceId>&& mat) noexcept;

        inline column_view<DenseMat, Type, DeviceId>&
        operator=(std::initializer_list<Type> const& init);

        template<typename DenseVec>
        inline column_view<DenseMat, Type, DeviceId>&
        operator=(dense_vector< DenseVec, Type, column, DeviceId> const& mat);

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

        inline size_t const&
        shape() const noexcept;
    };
}

#include <mgcpp/matrix/column_view.tpp>
#endif
