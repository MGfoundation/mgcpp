
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

#include <mgcpp/cublas/cublas_fft.hpp>

namespace mgcpp
{
    template<typename DeviceVec,
             typename Type,
             allignment Align,
             size_t DeviceId>
    inline device_vector<Type, Align, DeviceId,
                         typename DeviceVec::allocator_type>
    rfft(dense_vector<DeviceVec, Type, Align, DeviceId> const& vec)
    {
        using allocator_type = typename DeviceVec::allocator_type;

        auto const& dev_vec = ~vec;

        size_t n = dev_vec.shape();

        auto result = device_vector<Type, Align, DeviceId, allocator_type>(n / 2 * 2 + 2);

        auto status = mgcpp::cublas_rfft(n, dev_vec.data(), result.data_mutable());
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }
}
