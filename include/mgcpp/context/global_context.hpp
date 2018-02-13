
//          Copyright RedPortal, mujjingun 2017 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_CONTEXT_GLOBAL_CONTEXT_HPP_
#define _MGCPP_CONTEXT_GLOBAL_CONTEXT_HPP_

#include <initializer_list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <mgcpp/context/thread_context.hpp>

namespace mgcpp {
class global_context {
 private:
  std::mutex _mtx;
  hash_table<std::thread::id, thread_context> _thread_ctx;
  hash_table<std::thread::id, int> _context_ref_cnt;

  void reference_cnt_incr();

 public:
  global_context() = default;

  global_context(global_context const& other) = delete;
  global_context(global_context&& other) = delete;
  global_context& operator=(global_context const& other) = delete;
  global_context& operator=(global_context&& other) = delete;

  static global_context& global_context_setting();

  static void reference_cnt_decr();

  static thread_context& get_thread_context();
};
}  // namespace mgcpp
#endif
