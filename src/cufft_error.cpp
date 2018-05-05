
#include <mgcpp/system/cufft_error.hpp>

#include <cuda_runtime_api.h>

namespace mgcpp {
class cufft_error_category_t : public std::error_category {
 public:
  const char* name() const noexcept override;

  std::string message(int ev) const override;
} cufft_error_category;

const char* cufft_error_category_t::name() const noexcept {
  return "cufft";
}

std::string cufft_error_category_t::message(int ev) const {
  switch (static_cast<cufft_error_t>(ev)) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS: The cuFFT operation was successful.";

    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN: cuFFT was passed an invalid plan handle.";

    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED: cuFFT failed to allocate GPU or CPU memory";

    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE: No longer used.";

    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE: User specified an invalid pointer or "
             "parameter.";

    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR: Driver or internal cuFFT library error.";

    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED: Failed to execute an FFT on the GPU.";

    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED: The cuFFT library failed to initialize.";

    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE: User specified an invalid transform size.";

    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA: No longer used.";

    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST: Missing parameters in call.";

    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE: Execution of a plan was on different GPU "
             "than plan creation.";

    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR: Internal plan database error.";

    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE: No workspace has been provided prior to plan "
             "execution.";

    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED: Function does not implement functionality "
             "for parameters given.";

    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR: Used in previous versions.";

#if CUDART_VERSION >= 8000 // added in CUDA toolkit v8.0
    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED: Operation is not supported for parameters "
             "given.";
#endif
  }
  return "";
}
}  // namespace mgcpp

std::error_code make_error_code(mgcpp::cufft_error_t err) noexcept {
  return {static_cast<int>(err), mgcpp::cufft_error_category};
}
