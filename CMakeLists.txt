
#              Copyright RedPortal, mujjingun 2017 - 2018.
#  Distributed under the Boost Software License, Version 1.0.
#        (See accompanying file LICENSE or copy at
# 	        http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.8)
project(mgcpp)

macro(mgcpp_display_elements
	MGCPP_DISPLAY_MESSAGE
	MGCPP_LIST_TO_DISPLAY)
    message("[ ${MGCPP_DISPLAY_MESSAGE} ]")
    foreach(_MGCPP_DISPLAY_ELEMENT ${MGCPP_LIST_TO_DISPLAY})
	message(STATUS ${_MGCPP_DISPLAY_ELEMENT})
    endforeach()
endmacro()

# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ${CMAKE_SOURCE_DIR}/lib)

# cmake modules path
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# options && command line arguments
option(USE_MAGMA "compile mgcpp with magma features" ON)
option(USE_HALF "compile mgcpp with half precision floating point features" ON)
option(BUILD_TESTING "compile mgcpp with tests" ON)
option(BUILD_DOC "Build documentation" ON)
option(BUILD_BENCHMARK "Build library performance benchmark" OFF)
option(BUILD_EXAMPLES "Build API example code" OFF)

# cuda dependency
find_package(CUDA REQUIRED)

if(NOT CUDA_HOST_COMPILER)
    message(STATUS "separate cuda host compiler  wasn't provided, using default")
endif()

if(NOT CUDA_ARCH)
    message(STATUS "cuda arch wasn't provided, running feature test")
    include(feature_test)
else()
    set(CUDA_GEN_CODE "-gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
endif()

if(WIN32)
    set(CUDA_NVCC_FLAGS "")
else()
    set(CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
endif()

if(USE_HALF)
    if(CUDA_ARCH LESS "53")
	message(FATAL_ERROR "cuda devices lower than 5.3 do not support half precision type")
    endif()
    add_definitions(-DUSE_HALF)
endif()


# project source files
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(KERNELS_SOURCE_DIR ${SOURCE_DIR}/kernels)

set(BASE_SOURCE_FILES
    ${KERNELS_SOURCE_DIR}/hadamard.cu
    ${KERNELS_SOURCE_DIR}/map.cu
    ${KERNELS_SOURCE_DIR}/reduce.cu
    ${KERNELS_SOURCE_DIR}/fill.cu
    ${KERNELS_SOURCE_DIR}/convert.cu)

set(MAIN_SOURCE_FILES 
    ${SOURCE_DIR}/init.cpp
    ${SOURCE_DIR}/global_context.cpp
    ${SOURCE_DIR}/cublas_error.cpp
    ${SOURCE_DIR}/cufft_error.cpp
    ${SOURCE_DIR}/cuda_error.cpp
    ${SOURCE_DIR}/error_code.cpp
    ${SOURCE_DIR}/mgblas_error.cpp
    ${SOURCE_DIR}/thread_context.cpp
    ${SOURCE_DIR}/expression.cpp
    ${SOURCE_DIR}/eval_cache.cpp
    ${SOURCE_DIR}/new_delete_resource.cpp
    ${SOURCE_DIR}/cudamalloc_resource.cpp
    ${SOURCE_DIR}/new_delete_resource.cpp
    ${SOURCE_DIR}/device_memory_resource.cpp
    ${SOURCE_DIR}/cuda_libs/cublas.cpp
    ${SOURCE_DIR}/cuda_libs/cufft_fft.cpp
    ${SOURCE_DIR}/cuda/device.cpp)

mgcpp_display_elements("Source files for ${PROJECT_NAME}base"
    "${BASE_SOURCE_FILES}")

cuda_add_library("${PROJECT_NAME}base" STATIC ${BASE_SOURCE_FILES})

set(CUDA_VERBOSE_BUILD ON)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; ${CUDA_GEN_CODE}")

message("build flags = ${CUDA_NVCC_FLAGS}")

mgcpp_display_elements("Source files for ${PROJECT_NAME}"
    "${MAIN_SOURCE_FILES}")

add_library(${PROJECT_NAME} STATIC SHARED ${MAIN_SOURCE_FILES})

if(WIN32 AND MSVC)
    set(BUILD_FLAGS
	"${CMAKE_CXX_FLAGS} /W3 /MP")
else()
    if(WIN32)
	set(BUILD_FLAGS
	    "${CMAKE_CXX_FLAGS} -g -Wall -Wextra")
    else()
	set(BUILD_FLAGS
            "${CMAKE_CXX_FLAGS} -g -fPIC -Wall -Wextra")
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(BUILD_FLAGS
            "${BUILD_FLAGS} -Wno-mismatched-tags")
    endif()
endif()
message("C++ flags = ${BUILD_FLAGS}")

set_target_properties(${PROJECT_NAME}
    PROPERTIES COMPILE_FLAGS "${BUILD_FLAGS}"
               CXX_STANDARD 14
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF)

# Include directories
target_include_directories("${PROJECT_NAME}base"
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

set(CUDA_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include)


## mgcpp dependencies
set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/third_party)

# mgcpp base
target_link_libraries(${PROJECT_NAME}
    "${PROJECT_NAME}base")

set(MGCPP_TARGETS ${MGCPP_TARGETS}
    "${PROJECT_NAME}base")

# CUDA
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CUDA_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES})

# cuBLAS
cuda_add_cublas_to_target("${PROJECT_NAME}")

# cuFFT
cuda_add_cufft_to_target("${PROJECT_NAME}")

# cuSPARSE
target_link_libraries(${PROJECT_NAME}
    ${CUDA_cusparse_LIBRARIES})

# outcome 
find_package(Boost COMPONENTS system REQUIRED)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${THIRD_PARTY_DIR}/boost-outcome/include)

target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES})

# half
if(USE_HALF)
    set(HALF_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/third_party/half)
    target_include_directories(${PROJECT_NAME}
	PUBLIC
	${HALF_INCLUDE_DIRS})
endif()

# cuBLAS, MAGMA dependency
if(USE_MAGMA)
    find_package(MAGMA REQUIRED)
    message(STATUS "Magma found, building with Magma")
    target_include_directories(${PROJECT_NAME}
	PUBLIC
	${MAGMA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}
	PUBLIC
	${MAGMA_LIBRARY_DIRS})
else()
    message(STATUS "Not building with Magma (certain features will be restricted)")
endif(USE_MAGMA)

set(MGCPP_TARGETS
    ${PROJECT_NAME})

# Tests
if(BUILD_TESTING)
    enable_testing()

    # find gtest dependency
    find_package(GTest REQUIRED)

    # test source files
    set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)
    add_subdirectory(${TEST_DIR})

    set(MGCPP_BINARIES ${MGCPP_BINARIES} ${TEST_DIR}/${PROJECT_NAME}_test)
endif()

# Tests
if(BUILD_BENCHMARK)
    # find google benchmark dependency
    find_package(benchmark REQUIRED)

    set(BENCHMARK_DIR ${PROJECT_SOURCE_DIR}/benchmark)
    add_subdirectory(${BENCHMARK_DIR})

    set(MGCPP_BINARIES ${MGCPP_BINARIES} ${TEST_DIR}/${PROJECT_NAME}_bench)
endif()

# Build Doxygen
if (BUILD_DOC)
    # check if Doxygen is installed
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        # set input and output files
        set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        # request to configure the file
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
	message("Doxygen build started")

        # note the option ALL which allows to build the docs together with the application
        add_custom_target(doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)

    else (DOXYGEN_FOUND)
	message("Doxygen need to be installed to generate the doxygen documentation")
    endif (DOXYGEN_FOUND)
endif (BUILD_DOC)

# Build Example code
if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_EXAMPLES)

# install
install(TARGETS ${MGCPP_TARGETS}
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
    ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib
    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib)

install(FILES ${MGCPP_BINARIES}
    DESTINATION ${PROJECT_SOURCE_DIR}/bin)
