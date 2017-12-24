
#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

#include <cufft.h>
#include <mgcpp/system/error_code.hpp>

namespace mgcpp
{
    template<>
    inline outcome::result<void>
    cublas_rfft(size_t n, float const* x, float* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_R2C, 1);
        if (status != status_t::success) return status;

        status = cufftExecR2C(plan,
                              reinterpret_cast<cufftReal*>(const_cast<float*>(x)),
                              reinterpret_cast<cufftComplex*>(result));
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    template<>
    inline outcome::result<void>
    cublas_rfft(size_t n, double const* x, double* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_D2Z, 1);
        if (status != status_t::success) return status;

        status = cufftExecD2Z(plan,
                              reinterpret_cast<cufftDoubleReal*>(const_cast<double*>(x)),
                              reinterpret_cast<cufftDoubleComplex*>(result));
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    template<>
    inline outcome::result<void>
    cublas_irfft(size_t n, float const* x, float* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_C2R, 1);
        if (status != status_t::success) return status;

        status = cufftExecC2R(plan,
                              reinterpret_cast<cufftComplex*>(const_cast<float*>(x)),
                              reinterpret_cast<cufftReal*>(result));
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    template<>
    inline outcome::result<void>
    cublas_irfft(size_t n, double const* x, double* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_Z2D, 1);
        if (status != status_t::success) return status;

        status = cufftExecZ2D(plan,
                              reinterpret_cast<cufftDoubleComplex*>(const_cast<double*>(x)),
                              reinterpret_cast<cufftDoubleReal*>(result));
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;

        return outcome::success();
    }

    template<>
    inline outcome::result<void>
    cublas_cfft(size_t n, float const* x, float* result, cublas::fft_direction direction)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_C2C, 1);
        if (status != status_t::success) return status;

        status = cufftExecC2C(plan,
                              reinterpret_cast<cufftComplex*>(const_cast<float*>(x)),
                              reinterpret_cast<cufftComplex*>(result),
                              static_cast<int>(direction));
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    template<>
    inline outcome::result<void>
    cublas_cfft(size_t n, double const* x, double* result, cublas::fft_direction direction)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);
        if (status != status_t::success) return status;

        status = cufftExecZ2Z(plan,
                              reinterpret_cast<cufftDoubleComplex*>(const_cast<double*>(x)),
                              reinterpret_cast<cufftDoubleComplex*>(result),
                              static_cast<int>(direction));
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }
}
