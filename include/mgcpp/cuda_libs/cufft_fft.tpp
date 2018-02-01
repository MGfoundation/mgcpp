
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/global/complex.hpp>

#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

#include <cufft.h>
#include <mgcpp/system/error_code.hpp>

namespace mgcpp
{
    inline outcome::result<void>
    cufft_rfft(size_t n, float const* x, cuComplex* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_R2C, 1);
        if (status != status_t::success) return status;

        status = cufftExecR2C(plan,
                              const_cast<float*>(x),
                              result);
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    inline outcome::result<void>
    cufft_rfft(size_t n, double const* x, cuDoubleComplex* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_D2Z, 1);
        if (status != status_t::success) return status;

        status = cufftExecD2Z(plan,
                              const_cast<double*>(x),
                              result);
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    inline outcome::result<void>
    cufft_irfft(size_t n, cuComplex const* x, float* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_C2R, 1);
        if (status != status_t::success) return status;

        status = cufftExecC2R(plan,
                              const_cast<cuComplex*>(x),
                              result);
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    inline outcome::result<void>
    cufft_irfft(size_t n, cuDoubleComplex const* x, double* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_Z2D, 1);
        if (status != status_t::success) return status;

        status = cufftExecZ2D(plan,
                              const_cast<cuDoubleComplex*>(x),
                              result);
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;

        return outcome::success();
    }

    inline outcome::result<void>
    cufft_cfft(size_t n, cuComplex const* x, cuComplex* result, cufft::fft_direction direction)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_C2C, 1);
        if (status != status_t::success) return status;

        status = cufftExecC2C(plan,
                              const_cast<cuComplex*>(x),
                              result,
                              static_cast<int>(direction));
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    inline outcome::result<void>
    cufft_cfft(size_t n, cuDoubleComplex const* x, cuDoubleComplex* result, cufft::fft_direction direction)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);
        if (status != status_t::success) return status;

        status = cufftExecZ2Z(plan,
                              const_cast<cuDoubleComplex*>(x),
                              result,
                              static_cast<int>(direction));
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    inline outcome::result<void>
    cufft_rfft2(size_t n, size_t m, float const* x, cuComplex* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan2d(&plan, m, n, CUFFT_R2C);
        if (status != status_t::success) return status;

        status = cufftExecR2C(plan,
                              const_cast<float*>(x),
                              result);
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    inline outcome::result<void>
    cufft_rfft2(size_t n, size_t m, double const* x, cuDoubleComplex* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan2d(&plan, m, n, CUFFT_D2Z);
        if (status != status_t::success) return status;

        status = cufftExecD2Z(plan,
                              const_cast<double*>(x),
                              result);
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    inline outcome::result<void>
    cufft_irfft2(size_t n, size_t m, cuComplex const* x, float* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan2d(&plan, m, n, CUFFT_C2R);
        if (status != status_t::success) return status;

        status = cufftExecC2R(plan,
                              const_cast<cuComplex*>(x),
                              result);
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    inline outcome::result<void>
    cufft_irfft2(size_t n, size_t m, cuDoubleComplex const* x, double* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan2d(&plan, m, n, CUFFT_Z2D);
        if (status != status_t::success) return status;

        status = cufftExecZ2D(plan,
                              const_cast<cuDoubleComplex*>(x),
                              result);
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }

    inline outcome::result<void>
    cufft_cfft2(size_t n, size_t m, cuComplex const* x, cuComplex* result, cufft::fft_direction direction)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan2d(&plan, m, n, CUFFT_C2C);
        if (status != status_t::success) return status;

        status = cufftExecC2C(plan,
                              const_cast<cuComplex*>(x),
                              result,
                              static_cast<int>(direction));
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }


    inline outcome::result<void>
    cufft_cfft2(size_t n, size_t m, cuDoubleComplex const* x, cuDoubleComplex* result, cufft::fft_direction direction)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan2d(&plan, m, n, CUFFT_Z2Z);
        if (status != status_t::success) return status;

        status = cufftExecZ2Z(plan,
                              const_cast<cuDoubleComplex*>(x),
                              result,
                              static_cast<int>(direction));
        if (status != status_t::success) return status;

        status = cufftDestroy(plan);
        if (status != status_t::success) return status;
        return outcome::success();
    }
}
