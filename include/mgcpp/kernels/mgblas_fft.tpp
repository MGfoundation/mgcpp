
#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

#include <cufft.h>

namespace mgcpp
{
    inline outcome::result<void>
    mgblas_rfft(size_t n, float* const x, float* result)
    {
        cufftResult result;
        cufftHandle plan;

        result = cufftPlan1d(&plan, n, CUFFT_R2C, 1);
        if (result != CUFFT_SUCCESS) {
            return result;
        }

        result = cufftExecR2C(plan,
                              reinterpret_cast<cufftReal*>(x),
                              reinterpret_cast<cufftComplex*>(result));
        if (reuslt != CUFFT_SUCCESS) {
            return result;
        }

        //cufftDestroy(plan);
        return outcome::success();
    }
}
