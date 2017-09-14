#include <iostream>
#include <thread>

#include <mgcpp/cuda/cuda_template_stdlib.hpp>

int main()
{
    float* ptr = mgcpp::cuda_malloc<float>(5);
    
    return 0;
}
