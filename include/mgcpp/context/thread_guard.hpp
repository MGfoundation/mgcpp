
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CONTEXT_THREAD_GUARD_HPP_
#define _MGCPP_CONTEXT_THREAD_GUARD_HPP_

#include <cstdlib>
#include <initializer_list>

namespace mgcpp
{
    class thread_guard
    {
    public:
        inline explicit
        thread_guard(std::initializer_list<size_t> device, bool cublas);
        
        inline explicit
        thread_guard(size_t device, bool cublas);

        inline ~thread_guard();
    };
}

#include <mgcpp/context/thread_guard.ipp>
#endif
