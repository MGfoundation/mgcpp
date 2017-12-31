//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cublas/cufft_fft.hpp>
#include <mgcpp/operations/fft.hpp>
#include <mgcpp/operations/mult.hpp>
#include <mgcpp/vector/dense_vector.hpp>
#include <mgcpp/vector/device_vector.hpp>

namespace mgcpp
{
    template<typename DeviceVec,
             typename Type,
             alignment Align,
             size_t DeviceId>
    decltype(auto)
    strict::
    rfft(dense_vector<DeviceVec, Type, Align, DeviceId> const& vec)
    {
        using allocator_type = typename DeviceVec::allocator_type;
        using result_allocator_type =
            typename allocator_type::template rebind_alloc<complex<Type>>;

        auto const& dev_vec = ~vec;

        size_t fft_size = dev_vec.shape();
        size_t output_size = fft_size / 2 + 1;

        auto result = device_vector<complex<Type>,
                                    Align,
                                    DeviceId,
                                    result_allocator_type>(output_size);

        auto status = mgcpp::cublas_rfft(fft_size,
                                         dev_vec.data(),
                                         result.data_mutable());
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }

    template<typename DeviceVec,
             typename Type,
             alignment Align,
             size_t DeviceId>
    decltype(auto)
    strict::
    irfft(dense_vector<DeviceVec, complex<Type>, Align, DeviceId> const& vec, int n)
    {
        using allocator_type = typename DeviceVec::allocator_type;
        using result_allocator_type =
            typename allocator_type::template rebind_alloc<Type>;

        auto const& dev_vec = ~vec;

        size_t fft_size = n;
        if (n < 0)
            fft_size = (dev_vec.shape() - 1) * 2;
        else if (fft_size / 2 + 1 > dev_vec.shape())
        {
            // FIXME: zero-pad input to length floor(n/2)+1
            MGCPP_THROW_RUNTIME_ERROR("Zero-pad FFT unimplemented");
        }
        size_t output_size = fft_size;

        auto result = device_vector<Type,
                                    Align,
                                    DeviceId,
                                    result_allocator_type>(output_size);

        auto status = mgcpp::cublas_irfft(fft_size,
                                          dev_vec.data(),
                                          result.data_mutable());
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        // Normalize the result
        result = mgcpp::strict::mult(static_cast<Type>(1. / fft_size), result);
        return result;
    }


    template<typename DeviceVec,
             typename Type,
             alignment Align,
             size_t DeviceId>
    decltype(auto)
    strict::
    cfft(dense_vector<DeviceVec, complex<Type>, Align, DeviceId> const& vec,
         fft_direction direction)
    {
        using allocator_type = typename DeviceVec::allocator_type;
        using result_allocator_type =
            typename allocator_type::template rebind_alloc<complex<Type>>;

        auto const& dev_vec = ~vec;

        size_t fft_size = dev_vec.shape();
        size_t output_size = fft_size;


        auto result = device_vector<complex<Type>,
                                    Align,
                                    DeviceId,
                                    result_allocator_type>(output_size);

        cublas::fft_direction dir;
        if (direction == fft_direction::forward)
            dir = cublas::fft_direction::forward;
        else
            dir = cublas::fft_direction::inverse;

        auto status = mgcpp::cublas_cfft(fft_size, dev_vec.data(),
                                         result.data_mutable(),
                                         dir);
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        // Normalize the result
        if (direction == fft_direction::inverse)
            result = mgcpp::strict::mult(static_cast<Type>(1. / fft_size),
                                         result);

        return result;
    }

    template<typename DeviceMat,
                typename Type,
                size_t DeviceId>
    decltype(auto)
    strict::
    rfft(dense_matrix<DeviceMat, Type, DeviceId> const& mat)
    {
        using allocator_type = typename DeviceMat::allocator_type;
        using result_allocator_type =
            typename change_allocator_type<allocator_type, complex<Type>>::type;

        auto const& dev_mat = ~mat;

        auto fft_size = dev_mat.shape();
        auto output_size = std::make_pair(fft_size.first / 2 + 1, fft_size.second / 2 + 1);

        auto result = device_matrix<complex<Type>,
                                    DeviceId,
                                    result_allocator_type>(output_size.first, output_size.second);

        auto status = mgcpp::cublas_rfft2(fft_size.first, fft_size.second,
                                         dev_mat.data(),
                                         result.data_mutable());
        if(!status)
        { MGCPP_THROW_SYSTEM_ERROR(status.error()); }

        return result;
    }
}
