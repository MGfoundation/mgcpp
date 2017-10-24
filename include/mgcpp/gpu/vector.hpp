
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_GPU_VECTOR_HPP_
#define _MGCPP_GPU_VECTOR_HPP_

#include <cstdlib>

#include <mgcpp/cpu/forward.hpp>
#include <mgcpp/global/allignment.hpp>

namespace mgcpp
{
    namespace gpu
    {
        template<typename T,
                 size_t DeviceId = 0,
                 allignment Allign = row>
        class vector
        {
        private:
            T* _data;
            thread_context* _context;
            size_t _size;
            bool _released;

        public:
            inline vector() noexcept;

            inline ~vector() noexcept;

            inline vector(size_t size);

            inline vector(size_t size, T init);

            inline
            vector(gpu::vector<T, DeviceId, Allign> const& other);

            inline
            vector(gpu::vector<T, DeviceId, Allign>&& other) noexcept;

            inline gpu::vector<T, DeviceId, Allign>&
            operator=(gpu::vector<T, DeviceId, Allign> const& other);

            inline gpu::vector<T, DeviceId, Allign>&
            operator=(gpu::vector<T, DeviceId, Allign>&& other) noexcept;

            inline void 
            copy_from_host(cpu::vector<T, Allign>) const;

            template<size_t Xdim, size_t Ydim>
            inline cpu::vector<T, Allign>
            copy_to_host() const;

            inline T
            check_value(size_t i) const;

            inline T const*
            get_data() const noexcept;

            inline T*
            get_data_mutable() noexcept;

            inline T*
            release_data() noexcept;

            inline thread_context*
            get_thread_context() const noexcept;

            inline size_t
            shape() const noexcept;

            inline size_t
            size() const noexcept;
        }
    }
}

#include <mgcpp/gpu/vector.tpp>
#endif
