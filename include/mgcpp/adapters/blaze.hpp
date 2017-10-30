
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_ADAPTERS_BLAZE_HPP_
#define _MGCPP_ADAPTERS_BLAZE_HPP_

#include <mgcpp/adapters/adapter_base.hpp>

namespace blaze
{
    template<typename Type, bool SO> 
    class DynamicMatrix;

    template<typename Type, bool TF> 
    class DynamicVector;
}

namespace mgcpp
{
    template<typename Type, bool SO>
    struct adapter<blaze::DynamicMatrix<Type, SO>>
        : std::true_type
    {
        void
        operator()(blaze::DynamicMatrix<Type, SO> const& mat,
                   Type** out_p, size_t* m, size_t* n)
        {
            *out_p = mat.data();
            if(SO)
            { *m = mat.rows(); }
            else 
            { *n = mat.rows(); }
        }
    };

    template<typename Type, bool TF>
    struct adapter<blaze::DynamicVector<Type, TF>>
        : std::true_type
    {
        void
        operator()(blaze::DynamicMatrix<Type, TF> const& mat,
                   Type** out_p, size_t* size)
        {
            *out_p = mat.data();
            *size = mat.size();
        }
    };
}

#endif
