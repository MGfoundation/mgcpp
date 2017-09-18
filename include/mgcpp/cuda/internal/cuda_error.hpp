#ifndef _CUDA_ERROR_HPP_
#define _CUDA_ERROR_HPP_

namespace mgcpp
{
    namespace internal
    {
        enum class cuda_error_t
        {
            success = 0,
            memory_allocation = 2,
            initialization_error = 3,
            launch_failure = 4,
            prior_launch_failure = 5,
            launch_timeout = 6,
            launch_out_of_resources = 7,
            invalid_device_function = 8,
            invalid_configuration = 9,
            invalid_device = 10,
            invalid_value = 11,
            invalid_pitch_value = 12,
            invalid_symbol = 13,
            map_buffer_object_failed = 14,
            invalid_device_pointer = 17
        }; 

        const char* cuda_error_string(cuda_error_t err);
    }
}

#endif
