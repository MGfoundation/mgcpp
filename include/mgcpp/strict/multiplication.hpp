
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_STRICT_MULTIPLICATION_HPP_
#define _MGCPP_STRICT_MULTIPLICATION_HPP_

namespace mgcpp
{
    namespace strict 
    {
        template<typename ElemType, size_t DeviceId>
        gpu::matrix
        mult(gpu::matrix<ElemType, DeviceId> const& first,
             gpu::matrix<ElemType, DeviceId> const& second);

        template<typename ElemType,
                 size_t DeviceIdFirst, size_t DeviceIdSecond>
        gpu::matrix
        mult(gpu::matrix<ElemType, DeviceIdFirst> const& first,
             gpu::matrix<ElemType, DeviceIdSecond> const& second);

        template<typename ElemType, size_t DeviceId>
        void
        mult_assign(gpu::matrix<ElemType, DeviceId>& first,
                    gpu::matrix<ElemType, DeviceId> const& second);

        template<typename ElemType,
                 size_t DeviceIdFirst, size_t DeviceIdSecond>
        void
        mult_assign(gpu::matrix<ElemType, DeviceIdFirst>& first,
                    gpu::matrix<ElemType, DeviceIdSecond> const& second);
    }
}

#endif
