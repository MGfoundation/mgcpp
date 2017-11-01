
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <initializer_list>

#include <mgcpp/context/thread_guard.hpp>
#include <mgcpp/context/global_context.hpp>

namespace mgcpp
{
    thread_guard::
    thread_guard(std::initializer_list<size_t> device,
                 bool cublas)
    {
        auto ctx = global_context::get_thread_context();

        if(cublas)
        {
            for(auto i : device)
            {
                (void)ctx->get_cublas_context(device);
            }
        }
    }

    thread_guard::
    thread_guard(size_t device, bool cublas)
    {
        auto ctx = global_context::get_thread_context();

        if(cublas)
        {
            (void)ctx->get_cublas_context(device);
        }
    }

    thread_guard::
    ~thread_guard();
    {
        global_context::reference_cnt_decr();
    }
}

#endif
