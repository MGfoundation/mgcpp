
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <mgcpp/context/global_context.hpp>

namespace mgcpp
{
    global_context _singl_context{};

    thread_context&
    global_context::
    get_thread_context()
    {
        auto this_thread_id = std::this_thread::get_id();
        auto lck = std::unique_lock<std::mutex>(_singl_context._mtx);
        ++_singl_context._context_ref_cnt[this_thread_id];
        return _singl_context._thread_ctx[this_thread_id];
    }

    void
    global_context::
    reference_cnt_decr()
    {
        auto lck = std::unique_lock<std::mutex>(_singl_context._mtx);
        auto this_thread_id = std::this_thread::get_id();
        auto& ref = _singl_context._context_ref_cnt[this_thread_id];
        --ref;

        if(ref == 0)
            _singl_context._thread_ctx.erase(this_thread_id);
    }
}
