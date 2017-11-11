
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/kernels/mgblas_lv1.hpp>
#include <mgcpp/operations/hdmd.hpp>
#include <mgcpp/system/assert.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    template<typename T, size_t Device, allignment Allign, typename Alloc>
    inline device_vector<T, Device, Allign, Alloc>
    strict::
    hdmd(device_vector<T, Device, Allign, Alloc> const& first,
         device_vector<T, Device, Allign, Alloc> const& second)
    {
        MGCPP_ASSERT(first.shape() == second.shape(),
                     "matrix dimensions didn't match");

        size_t size = first.shape();
        device_vector<T, Device, Allign, Alloc> result(size);

        auto status = mgblas_vhp(first.data(), second.data(),
                                 result.data_mutable(),
                                 size);

        if(!status)
            MGCPP_THROW_SYSTEM_ERROR(status.error());

        return result;
    }
}
