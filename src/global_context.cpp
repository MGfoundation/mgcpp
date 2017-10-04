
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/global_context.hpp>

#include <cstdio>

namespace mgcpp
{
    global_context _singl_context;

    thread_context&
    global_contex::
    get_thread_context()
    {
        auto this_thread_id = std::this_thread::get_id();
        auto lck = std::unique_lock<std::mutex>(_mtx);
        ++_context_ref_cnt[id];
        return _singl_context._thread_ctx[this_thread_id];
    }

    void
    global_contex::
    reference_cnt_decr(std::thread::id const& id)
    {
        auto lck = std::unique_lock<std::mutex>(_mtx);
        --_context_ref_cnt[id];

        if(_context_ref_cnt == 0)
            _thread_ctx.erase(id);
    }

    // cublasHandle_t
    // thread_context::
    // get_cublas(size_t device_id) 
    // {
    //     return _device_managers[device_id].get_cublas();
    // }
}
