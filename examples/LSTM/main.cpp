
#include <mgcpp/mgcpp.hpp>
#include <cstdlib>

template<size_t InSize,
         size_t HiddenSize>
class LSTM
{
    // all gates are fused into one matrix for performance 
    mgcpp::device_matrix<float> w;
    mgcpp::device_vector<float> b;
    
};

int main()
{
    
}
