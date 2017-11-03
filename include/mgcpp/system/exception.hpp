
//          Copyright RedPortal 2017 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MGCPP_SYSTEM_EXCEPTIONS_HPP_
#define _MGCPP_SYSTEM_EXCEPTIONS_HPP_

#include <stdexcept>

#ifndef MGCPP_ABORT_ON_ERROR
#define MGCPP_ABORT_ON_ERROR true
#endif

#ifndef MGCPP_ERROR_MESSAGE_HANDLER
#define MGCPP_ERROR_MESSAGE_HANDLER( ... )      \
    fprintf(stderr, __VA_ARGS__)
#endif

#ifndef mgcpp_error_check
#define mgcpp_error_check( EXP )                            \
    do{try{EXP;}catch(std::exception const& e){             \
            MGCPP_ERROR_MESSAGE_HANDLER(                    \
                "[mgcpp errror] %s in %s line %d\n",        \
                e.what(), __FILE__, __LINE__);              \
            if(MGCPP_ABORT_ON_ERROR) exit(1);}}while(false)
#endif

#ifdef ERROR_CHECK_EXCEPTION
#define MGCPP_THROW(EXCEPTION)                      \
    do{ MGCPP_ERROR_MESSAGE_HANDLER(                \
            "[mgcpp errror] %s in %s line %d\n",    \
            EXCEPTION.what(), __FILE__, __LINE__);  \
        throw EXCEPTION; }while(false)
#endif

#ifndef MGCPP_THROW
#define MGCPP_THROW(EXCEPTION) throw EXCEPTION
#endif


#ifndef MGCPP_THROW_SYSTEM_ERROR
#define MGCPP_THROW_SYSTEM_ERROR(ERROR, ...)    \
    MGCPP_THROW(std::system_error(ERROR))
#endif

#ifndef MGCPP_THROW_BAD_ALLOC
#define MGCPP_THROW_BAD_ALLOC                   \
    MGCPP_THROW(std::bad_alloc())
#endif

#ifndef MGCPP_THROW_INVALID_ARGUMENT
#define MGCPP_THROW_INVALID_ARGUMENT( MESSAGE )     \
    MGCPP_THROW(std::invalid_argument( MESSAGE ))
#endif

#ifndef MGCPP_THROW_LENGTH_ERROR
#define MGCPP_THROW_LENGTH_ERROR( MESSAGE )     \
    MGCPP_THROW(std::length_error( MESSAGE ))
#endif

#ifndef MGCPP_THROW_OUT_OF_RANGE
#define MGCPP_THROW_OUT_OF_RANGE( MESSAGE )     \
    MGCPP_THROW(std::out_of_range( MESSAGE ))
#endif

#ifndef MGCPP_THROW_RUNTIME_ERROR
#define MGCPP_THROW_RUNTIME_ERROR( MESSAGE )    \
    MGCPP_THROW(std::runtime_error( MESSAGE ))
#endif

#ifndef MGCPP_THROW_OVERFLOW_ERROR
#define MGCPP_THROW_OVERFLOW_ERROR( MESSAGE )   \
    MGCPP_THROW(std::overflow_error( MESSAGE ))
#endif

#ifndef MGCPP_THROW_UNDERFLOW_ERROR
#define MGCPP_THROW_UNDERFLOW_ERROR( MESSAGE )      \
    MGCPP_THROW(std::underflow_error( MESSAGE ))
#endif

#endif
