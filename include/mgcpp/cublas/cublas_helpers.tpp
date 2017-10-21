
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/cublas/cublas_helpers.hpp>
#include <mgcpp/system/error_code.hpp>
#include <mgcpp/system/exception.hpp>

namespace mgcpp
{
    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_set_matrix(size_t rows, size_t cols,
                      ElemType const* from_host, size_t ld_from, 
                      ElemType* to_gpu, size_t ld_to)
    {
        std::error_code status =
            cublasSetMatrix(rows, cols,
                            rows * cols * sizeof(ElemType),
                            from_host, ld_from, to_gpu, ld_to);

        if(status != status_t::success)
            return status;
        else
            return to_gpu; 
    }


    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_set_matrix(size_t rows, size_t cols,
                      ElemType const* from_host, ElemType* to_gpu)
    {
        std::error_code status =
            cublasSetMatrix(rows, cols,
                            rows * cols * sizeof(ElemType),
                            from_host, rows, to_gpu, rows);

        if(status != status_t::success)
            return status;
        else
        return to_gpu;
    }

    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_get_matrix(size_t rows, size_t cols,
                      ElemType const* from_gpu, size_t ld_from, 
                      ElemType* to_host, size_t ld_to)
    {
        std::error_code status =
            cublasGetMatrix(rows, cols, sizeof(ElemType),
                            from_gpu, ld_from, to_host, ld_to);

        if(status != status_t::success)
            return status;
        else
            return to_host; 
    }


    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_get_matrix(size_t rows, size_t cols,
                      ElemType const* from_gpu, ElemType* to_host)
    {
        std::error_code status =
            cublasGetMatrix(rows, cols,
                            rows * cols * sizeof(ElemType),
                            from_gpu, rows, to_host, rows);

        if(status != status_t::success)
            return status;
        else
            return to_host;
    }

    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_set_vector(size_t size,
                      ElemType const* from_host, size_t spacing_host,
                      ElemType* to_gpu, size_t spacing_gpu)
    {
        std::error_code status =
            cublasSetVector(size * sizeof(ElemType),
                            from_host, spacing_host,
                            to_gpu, spacing_gpu);

        if(status != status_t::success)
            return status;
        else
        return to_gpu;
    }


    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_set_vector(size_t size,
                      ElemType const* from_host, ElemType* to_gpu )
    {
        std::error_code status =
            cublasSetVector(size * sizeof(ElemType),
                            from_host, 1, to_gpu, 1);

        if(status != status_t::success)
            return status;
        else
            return to_gpu;
    }

    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_get_vector(size_t size,
                      ElemType const* from_gpu, size_t spacing_gpu,
                      ElemType* to_host, size_t spacing_host)
    {
        std::error_code status =
            cublasGetVector(size * sizeof(ElemType),
                            from_gpu, spacing_gpu,
                            to_host, spacing_host);

        if(status != status_t::success)
            return status;
        else
            return to_host;
    }


    template<typename ElemType>
    outcome::result<ElemType*>
    cublas_get_vector(size_t size,
                      ElemType const* from_gpu, ElemType* to_host)
    {
        std::error_code status =
            cublasGetVector(size * sizeof(ElemType),
                            from_gpu, 1, to_host, 1);

        if(status != status_t::success)
            return status;
        else
            return to_host;
    }
}
