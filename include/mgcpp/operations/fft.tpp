
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

namespace mgcpp
{
    template<typename DeviceVec,
             typename Type,
             allignment Align,
             size_t DeviceId>
    inline device_vector<Type, Align, DeviceId,
                         typename DeviceVec::allocator_type>
    fft(dense_vector<DeviceVec, Type, Align, DeviceId> const& vec)
    {
        using allocator_type = typename DeviceVec::allocator_type;

        auto dev_vec = ~vec;

        size_t n = dev_vec.shape();

        auto result = device_vector<Type, Align, DeviceId, allocator_type>(n);
        return result;
    }
}
