#ifndef _CUDA_EXCEPTIONS_HPP_
#define _CUDA_EXCEPTIONS_HPP_

#include <exception>

namespace mgcpp
{
    class cuda_bad_alloc
        : public std::exception
    {
    private:
        char const* _msg; 

    public:
        inline cuda_bad_alloc(char const* msg)
            : _msg(msg)
        {}
        
        virtual const char* what() const throw()
        { return _msg; }
    };

    class cuda_bad_dealloc
        : public std::exception
    {
    private:
        char const* _msg; 

    public:
        inline cuda_bad_dealloc(char const* msg)
            : _msg(msg)
        {}
        
        virtual const char* what() const throw()
        { return _msg; }
    };
}

#endif
