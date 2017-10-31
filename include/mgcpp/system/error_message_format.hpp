
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_SYSTEM_ERROR_MESSAGE_FORMAT_HPP_
#define _MGCPP_SYSTEM_ERROR_MESSAGE_FORMAT_HPP_

#ifndef MGCPP_ERROR_MESSAGE_HANDLER
#include <cstdio>
#define MGCPP_ERROR_MESSAGE_HANDLER(MESSAGE, ...)   \
    fprintf(stderr, MESSAGE, __VA_ARGS__)
#endif

#ifndef MGCPP_HANDLE_ERROR_MESSAGE
#include <string>

namespace mgcpp
{
    template<typename Args>
    inline std::string
    string_meta_concat_impl(Args const& args)
    { return std::string(args); }

    template<typename Head, typename... Args>
    inline std::string
    string_meta_concat_impl(Head const& head,
                            Args... args)
    { return std::string(head) + string_meta_concat_impl(args...); }

    template<typename... Args>
    inline char const*
    string_meta_concat(Args... args)
    { return string_meta_concat_impl(args...).c_str(); }
}

#define MGCPP_HANDLE_ERROR_MESSAGE(...) string_meta_concat(__VA_ARGS__)
#endif

#endif
