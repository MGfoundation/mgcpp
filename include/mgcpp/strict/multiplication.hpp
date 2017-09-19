#ifndef _MGCPP_STRICT_MULTIPLICATION_
#define _MGCPP_STRICT_MULTIPLICATION_

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
        mult_assign(gpu::matrix<ElemType, DeviceId> const& first,
                    gpu::matrix<ElemType, DeviceId> const& second);

        template<typename ElemType,
                 size_t DeviceIdFirst, size_t DeviceIdSecond>
        void
        mult_assign(gpu::matrix<ElemType, DeviceIdFirst> const& first,
                    gpu::matrix<ElemType, DeviceIdSecond> const& second);
    }
}

#endif
