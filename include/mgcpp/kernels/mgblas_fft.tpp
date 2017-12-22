
#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

#include <cufft.h>

namespace mgcpp
{
    inline outcome::result<void>
    mgblas_rfft(size_t n, float const* x, float* result)
    {
        cufftResult status;
        cufftHandle plan;

        status = cufftPlan1d(&plan, n, CUFFT_R2C, 1);
        if (status != CUFFT_SUCCESS) {
            if (status == CUFFT_ALLOC_FAILED) fprintf(stderr, "The allocation of GPU resources for the plan failed.\n");
            if (status == CUFFT_INVALID_VALUE) fprintf(stderr, "One or more invalid parameters were passed to the API.\n");
            if (status == CUFFT_INTERNAL_ERROR) fprintf(stderr, "An internal driver error was detected.\n");
            if (status == CUFFT_SETUP_FAILED) fprintf(stderr, "The cuFFT library failed to initialize.\n");
            if (status == CUFFT_INVALID_SIZE) fprintf(stderr, "The nx = %zu or batch parameter is not a supported size.\n", n);
            return mgcpp::kernel_status_t::invalid_range;
        }

        status = cufftExecR2C(plan,
                              reinterpret_cast<cufftReal*>(const_cast<float*>(x)),
                              reinterpret_cast<cufftComplex*>(result));
        if (status != CUFFT_SUCCESS) {
            if (status == CUFFT_INVALID_PLAN) fprintf(stderr, "The plan parameter is not a valid handle. \n");
            if (status == CUFFT_INVALID_VALUE) fprintf(stderr, "At least one of the parameters idata and odata is not valid.\n");
            if (status == CUFFT_INTERNAL_ERROR) fprintf(stderr, "An internal driver error was detected.\n");
            if (status == CUFFT_EXEC_FAILED) fprintf(stderr, "cuFFT failed to execute the transform on the GPU.\n");
            if (status == CUFFT_SETUP_FAILED) fprintf(stderr, "The cuFFT library failed to initialize.\n");
            return mgcpp::kernel_status_t::invalid_range;
        }

        status = cufftDestroy(plan);
        if (status != CUFFT_SUCCESS) {
            if (status == CUFFT_INVALID_PLAN) fprintf(stderr, "The plan parameter is not a valid handle. \n");
            return mgcpp::kernel_status_t::invalid_range;
        }
        return outcome::success();
    }
}
