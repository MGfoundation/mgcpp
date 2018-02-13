
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)


#include <mgcpp/operations/pad.hpp>
#include <mgcpp/kernels/mgblas_helpers.hpp>
#include <mgcpp/cuda/memory.hpp>

namespace mgcpp
{
    template<typename DenseVec,
             typename Type,
             size_t Device>
    inline decltype(auto)
    strict::
    pad(dense_vector<DenseVec, Type, Device> const& vec,
        pad_size_t pad,
        typename value_type<Type>::type pad_constant)
    {
        using allocator_type = typename DenseVec::allocator_type;

        auto dvec = ~vec;

        auto new_size = pad.first + dvec.size() + pad.second;

        if (pad.first == 0)
        {
            auto orig_size = dvec.size();
            dvec.resize(new_size);

            if (pad.second > 0)
            {
                auto fill_result = mgblas_fill(dvec.data_mutable() + orig_size,
                                               pad_constant,
                                               pad.second);
                if (!fill_result)
                { MGCPP_THROW_SYSTEM_ERROR(fill_result.error()); }
            }

            return dvec;
        }
        else
        {
            auto result = device_vector<Type,
                                        Device,
                                        allocator_type>(new_size, pad_constant);

            auto cpy_status = cuda_memcpy(result.data_mutable() + pad.first,
                                          dvec.data(),
                                          dvec.size(),
                                          cuda_memcpy_kind::device_to_device);
            if(!cpy_status)
            { MGCPP_THROW_SYSTEM_ERROR(cpy_status.error()); }

            if (pad.first > 0)
            {
                auto fill_result = mgblas_fill(result.data_mutable(),
                                            pad_constant,
                                            pad.first);
                if (!fill_result)
                { MGCPP_THROW_SYSTEM_ERROR(fill_result.error()); }
            }

            if (pad.second > 0)
            {
                auto fill_result = mgblas_fill(result.data_mutable() + pad.first + dvec.size(),
                                        pad_constant,
                                        pad.second);
                if (!fill_result)
                { MGCPP_THROW_SYSTEM_ERROR(fill_result.error()); }
            }

            return result;
        }

    }
}
