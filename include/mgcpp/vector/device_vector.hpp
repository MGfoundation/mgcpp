
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_VECTOR_DEVICE_VECTOR_HPP_
#define _MGCPP_VECTOR_DEVICE_VECTOR_HPP_

#include <mgcpp/adapters/adapters.hpp>
#include <mgcpp/allocators/default.hpp>
#include <mgcpp/context/thread_context.hpp>
#include <mgcpp/global/shape.hpp>
#include <mgcpp/system/concept.hpp>
#include <mgcpp/type_traits/device_value_type.hpp>
#include <mgcpp/type_traits/is_supported_type.hpp>
#include <mgcpp/vector/dense_vector.hpp>

#include <cstdlib>
#include <initializer_list>

namespace mgcpp {
template <typename Type, typename Alloc = mgcpp::default_allocator<Type>>
class device_vector : public dense_vector<device_vector<Type, Alloc>, Type> {
  static_assert(is_supported_type<Type>::value, "Element type not supported.");

 public:
  using this_type = device_vector<Type, Alloc>;
  using value_type = Type;
  using pointer = value_type*;
  using const_pointer = value_type const*;
  using device_value_type = typename device_value_type<Type>::type;
  using device_pointer = device_value_type*;
  using const_device_pointer = device_value_type const*;
  using result_type = this_type;
  template <typename T>
  using result_expr_type = dvec_expr<T>;
  using allocator_type = Alloc;
  using shape_type = mgcpp::shape<1>;
  using parent_type = dense_vector<device_vector<Type, Alloc>, Type>;

 private:
  thread_context* _context;
  size_t _shape;
  Alloc _allocator;
  device_pointer _data;
  size_t _capacity;

 public:
  inline device_vector() noexcept;

  inline ~device_vector() noexcept;

  inline explicit device_vector(Alloc const& alloc) noexcept;

  inline explicit device_vector(size_t size, Alloc const& alloc = Alloc());

  inline explicit device_vector(size_t size,
                                value_type init,
                                Alloc const& alloc = Alloc());

  inline explicit device_vector(size_t size,
                                const_pointer data,
                                Alloc const& alloc = Alloc());

  inline explicit device_vector(std::initializer_list<value_type> const& array,
                                Alloc const& alloc = Alloc());

  template <typename HostVec, MGCPP_CONCEPT(adapter<HostVec>::value)>
  inline explicit device_vector(HostVec const& host_mat,
                                Alloc const& alloc = Alloc());

  inline device_vector(device_vector<Type, Alloc> const& other);

  template <typename DenseVec>
  inline explicit device_vector(dense_vector<DenseVec, Type> const& other);

  inline device_vector(device_vector<Type, Alloc>&& other) noexcept;

  template <size_t S>
  static inline device_vector from_c_array(Type (&arr)[S],
                                           Alloc const& alloc = Alloc());

  template <typename DenseVec>
  inline device_vector<Type, Alloc>& operator=(
      dense_vector<DenseVec, Type> const& other);

  inline device_vector<Type, Alloc>& operator=(
      device_vector<Type, Alloc> const& other);

  inline device_vector<Type, Alloc>& operator=(
      device_vector<Type, Alloc>&& other) noexcept;

  inline device_vector<Type, Alloc>& zero();

  inline device_vector<Type, Alloc>& resize(
      size_t size,
      value_type pad_value = value_type{});

  inline void copy_to_host(pointer host_p) const;

  inline value_type check_value(size_t i) const;

  inline void set_value(size_t i, value_type value);

  inline const_device_pointer data() const noexcept;

  inline device_pointer data_mutable() noexcept;

  inline size_t capacity() const noexcept;

  inline device_pointer release_data() noexcept;

  inline thread_context* context() const noexcept;

  inline shape_type shape() const noexcept;

  inline Alloc allocator() const noexcept;

  inline size_t device_id() const noexcept;

  inline size_t size() const noexcept;
};
}  // namespace mgcpp

#include <mgcpp/vector/device_vector.tpp>
#endif
