
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
        if (status != CUFFT_SUCCESS) return status;

        status = cufftExecR2C(plan,
                              reinterpret_cast<cufftReal*>(const_cast<float*>(x)),
                              reinterpret_cast<cufftComplex*>(result));
        if (status != CUFFT_SUCCESS) return status;

        status = cufftDestroy(plan);
        if (status != CUFFT_SUCCESS) return status;
        return outcome::success();
    }

    template<>
    inline outcome::result<void>
    cublas_rfft(size_t n, double const* x, double* result)
    {
        std::error_code status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_D2Z, 1);
        if (status != CUFFT_SUCCESS) return status;

        status = cufftExecD2Z(plan,
                              reinterpret_cast<cufftDoubleReal*>(const_cast<double*>(x)),
                              reinterpret_cast<cufftDoubleComplex*>(result));
        if (status != CUFFT_SUCCESS) return status;

        status = cufftDestroy(plan);
        if (status != CUFFT_SUCCESS) return status;
        return outcome::success();
    }
}
