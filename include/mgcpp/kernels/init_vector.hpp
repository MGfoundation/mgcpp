#ifndef _MGCPP_GPU_INIT_VECTOR_HPP_
#define _MGCPP_GPU_INIT_VECTOR_HPP_

#include "../device/vector.hpp"

namespace mgcpp
{
  template<typename T>
  void init_device_vector(T** devPtr, size_t size);
}
  

#endif _MGCPP_GPU_INIT_VECTOR_HPP_
