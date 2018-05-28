
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_MATRIX_COLUMN_VIEW_HPP_
#define _MGCPP_MATRIX_COLUMN_VIEW_HPP_

#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/matrix/dense_matrix.hpp>
#include <mgcpp/vector/dense_vector.hpp>

#include <cstdlib>
#include <initializer_list>

namespace mgcpp {
template <typename DenseMat, typename Type>
class column_view : public dense_vector<column_view<DenseMat, Type>, Type> {
 public:
  using this_type = column_view<DenseMat, Type>;
  using value_type = Type;
  using result_type = this_type;
  using allocator_type = typename DenseMat::allocator_type;

 private:
  DenseMat* _matrix;
  size_t _column_idx;
  allocator_type _allocator;

 public:
  inline column_view() = delete;

  inline ~column_view() = default;

  inline column_view(dense_matrix<DenseMat, Type>& mat,
                     size_t i) noexcept;

  inline column_view(column_view<DenseMat, Type> const& other) =
      delete;

  inline column_view(column_view<DenseMat, Type>&& other) noexcept;

  inline column_view<DenseMat, Type>& operator=(
      column_view<DenseMat, Type> const& other);

  inline column_view<DenseMat, Type>& operator=(
      column_view<DenseMat, Type>&& other) noexcept;

  inline column_view<DenseMat, Type>& operator=(
      std::initializer_list<Type> const& init);

  template <typename DenseVec>
  inline column_view<DenseMat, Type>& operator=(
      dense_vector<DenseVec, Type> const& vec);

  inline void copy_to_host(Type* host_p) const;

  inline Type check_value(size_t i) const;

  inline Type const* data() const noexcept;

  inline Type* data_mutable() noexcept;

  inline thread_context* context() const noexcept;

  inline size_t shape() const noexcept;
};
}  // namespace mgcpp

#include <mgcpp/matrix/column_view.tpp>
#endif
