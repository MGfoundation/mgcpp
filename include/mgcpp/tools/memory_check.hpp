
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_TOOLS_MEMORY_CHECK_
#define _MGCPP_TOOLS_MEMORY_CHECK_

#include <cstdlib>

namespace mgcpp
{
    class leak_checker
    {
    private:
        size_t _device_id;
        size_t _before_free_memory;
        size_t _after_free_memory;
        bool _cached;

    public:
        leak_checker(size_t device_id = 0);

        operator bool() const noexcept;

        bool cache() noexcept;

        size_t initial_memory() const noexcept;
    };
}

#endif
