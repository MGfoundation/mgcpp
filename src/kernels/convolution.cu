#include <mgcpp/kernels/bits/convolution.cuh>

namespace mgcpp
{
    kernel_status_t
    mgblas_convolution(float* result, float const& f,
                       size_t dim_fx, size_t dim_fy,
                       float const& g)
    {
        mgblas_convolution(result, f, dim_fx, dim_fy, g);
        return; // what?
    }

    __global__ void
    mgblas_convolution_impl(float* result, float const& f,
                            size_t dim_fx, size_t dim_fy,
                            float const& g)
    {
        if(sizeof(g) != 9) throw exception();

        float* exf[(dim_fx+1)*(dim_fy+1)];
        add_padding(exf, f, dim_fx, dim_fy);

        remove_padding(f, exf, dim_fx, dim_fy);
    }

    void add_padding(float* destination, float* source,
                     size_t const& dim_fx, size_t const& dim_fy)
    {
        for(int i = 0; i < dim_fx+2; i++)
        {
            destination[i] = 0;
        }
        for(int i = 0; i < dim_fy; i++)
        {
            destination[(dim_fx+2)*(i+1)] = 0;
            destination[(dim_fx+3)*(i+1) - 1] = 0;
        }
        for(int i = 0; i < dim_fx+2; i++)
        {
            destination[(dim_fx+2)*(dim_fy+1) + i] = 0;
        }
    }

    void remove_padding(float* destination, float* source,
                        size_t const& dim_fx, size_t const& dim_fy)
    {
        for(int i = 0; i < dim_fy; i++)
        {
            for(int j = 0; j < dim_fx; j++)
            {
                destination[j+i*dim_fx] = source[j+(i+1)*(dim_fx+2)+1];
            }
        }
    }
}