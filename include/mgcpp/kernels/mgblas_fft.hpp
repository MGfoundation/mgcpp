
#ifndef _MGCPP_BLAS_FFT_HPP_
#define _MGCPP_BLAS_FFT_HPP_

#include <boost/outcome.hpp>
namespace outcome = BOOST_OUTCOME_V2_NAMESPACE;

namespace mgcpp
{
    template<typename T>
    inline outcome::result<void>
    mgblas_rfft(size_t n, T* const x, T* result);
}

#include <mgcpp/cublas/blas_fft.tpp>

#endif
