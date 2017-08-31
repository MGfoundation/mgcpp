cmake_minimum_required(VERSION 3.8)
project(mgcpp)

# options
option(USE_MAGMA "compile mgcpp with magma features" TRUE)

# cmake modules path
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# cuda dependency
find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS
    "-gencode arch=compute_30, code=sm_30; -std=c++11")

# project source files
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(SOURCE_FILES ${SOURCE_DIR}/main.cpp)

message("[ Source files for ${PROJECT_NAME} ]")
foreach(SOURCE_FILE ${SOURCE_FILES})
    message(STATUS "${SOURCE_FILE}")
endforeach()

cuda_add_library(${PROJECT_NAME} STATIC SHARED ${SOURCE_FILES})

## cmgcpp dependencies
# cuBLAS
cuda_add_cublas_to_target(${PROJECT_NAME})

# Build options
if(Win32 AND MSVC)
    set(BUILD_FLAGS
	"${CMAKE_CXX_FLAGS} /Ox /Wall /arch /MP")
else()
    set(BUILD_FLAGS
	"${CMAKE_CXX_FLAGS} -g -fPIC -Wall -Wextra")
endif()

message("C++ flags = ${BUILD_FLAGS}")

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    COMPILE_FLAGS "${BUILD_FLAGS}"
    CXX_STANDARD 11)

# Include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include)

# cuBLAS, MAGMA dependency
if(USE_MAGMA)
    find_package(MAGMA REQUIRED)
    target_include_directories(${PROJECT_NAME}
	PUBLIC
	${MAGMA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}
	PUBLIC
	${MAGMA_LIBRARY_DIRS})
endif(USE_MAGMA)

# ## Thread library dependency
# find_package(Threads REQUIRED)
# target_link_libraries(${PROJECT_NAME}
#     PUBLIC
#     ${CMAKE_THREAD_LIBS_INIT})

# install
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib)

