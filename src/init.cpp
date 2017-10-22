
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/global/init.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>
#include <mgcpp/cuda/device.hpp>

#include <cuda_runtime.h>

#include <iostream>

namespace mgcpp
{
    void
    init(bool print_system_info)
    {
        auto cuda_status = cuda_set_device(0);
        if(!cuda_status)
        {
            MGCPP_THROW_SYSTEM_ERROR(cuda_status.error());
        }

        if(print_system_info)
        {
            const int kb = 1024;
            const int mb = kb * kb;

            std::cout << "CUDA version: "
                      << CUDART_VERSION <<std::endl;    

            int devCount;
            std::error_code device_count_status
                = cudaGetDeviceCount(&devCount);
            if(device_count_status != status_t::success)
            {
                MGCPP_THROW_SYSTEM_ERROR(device_count_status);
            }

            std::cout << "Found "<< devCount
                      << " CUDA devices" << '\n' <<std::endl;

            for(int i = 0; i < devCount; ++i)
            {
                cudaDeviceProp props;
                std::error_code property_status =
                    cudaGetDeviceProperties(&props, i);
                if(!property_status)
                {
                    MGCPP_THROW_SYSTEM_ERROR(property_status);
                }

                std::cout << i << ": " << props.name
                          << ": " << props.major
                          << "." << props.minor << '\n';
                std::cout << "  Global memory: "
                          << props.totalGlobalMem / mb << "mb\n";
                std::cout << "  Shared memory: "
                          << props.sharedMemPerBlock / kb << "kb\n";
                std::cout << "  Constant memory: "
                          << props.totalConstMem / kb << "kb\n"; 
                std::cout << "  Block registers: "
                          << props.regsPerBlock
                          << '\n';
                std::cout << std::endl;
            }
        }
    }
}
