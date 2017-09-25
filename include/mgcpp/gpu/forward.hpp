#ifndef _MGCPP_GPU_FORWARD_HPP_
#define _MGCPP_GPU_FORWARD_HPP_


namespace mgcpp
{
    enum class allignment;
    
    namespace gpu
    {
        template<typename ElemType, size_t DeviceId>
        class matrix;

        template<typename ElemType,
                 allignment Allign,
                 size_t DeviceId>
        class vector;
    }
}

#endif
