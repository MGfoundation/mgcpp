
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)


#include <mgcpp/operations/pad.hpp>
#include <mgcpp/cuda/memory.hpp>

namespace mgcpp
{
    template<typename DenseVec,
             typename Type,
             size_t Device,
             alignment Align>
    inline decltype(auto)
    strict::
    pad(dense_vector<DenseVec, Type, Align, Device> const& vec,
        pad_size_t pad,
        typename value_type<Type>::type pad_constant)
    {
        using allocator_type = typename DenseVec::allocator_type;

        auto dvec = ~vec;

        auto new_size = pad.first + dvec.size() + pad.second;
        auto result = device_vector<Type,
                                    Align,
                                    Device,
                                    allocator_type>(new_size, pad_constant);

        cuda_memcpy(result.data_mutable() + pad.first, dvec.data(), dvec.size(), cuda_memcpy_kind::device_to_device);

        return result;
    }
}
