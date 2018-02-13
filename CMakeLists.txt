
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

if(USE_HALF AND (CUDA_ARCH LESS 53))
    message(FATAL_ERROR "cuda devices lower than 5.3 do not support half precision type")
elseif(USE_HALF)
    add_definitions(-DUSE_HALF)
endif()


# project source files
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(KERNELS_SOURCE_DIR ${SOURCE_DIR}/kernels)

set(BASE_SOURCE_FILES
    ${KERNELS_SOURCE_DIR}/hadamard.cu
    ${KERNELS_SOURCE_DIR}/absolute.cu
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
    ${SOURCE_DIR}/thread_context.cpp)

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
	"${CMAKE_CXX_FLAGS} /w3 /Ot /Ox /MP")
else()
    if(WIN32)
	set(BUILD_FLAGS
	    "${CMAKE_CXX_FLAGS} -std=c++14 -g -Wall -Wextra")
    else()
	set(BUILD_FLAGS
	    "${CMAKE_CXX_FLAGS} -std=c++14 -g -fPIC -Wall -Wextra")
    endif()
endif()
message("C++ flags = ${BUILD_FLAGS}")

set_target_properties(${PROJECT_NAME}
    PROPERTIES COMPILE_FLAGS "${BUILD_FLAGS}")


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
find_package(Boost REQUIRED)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${THIRD_PARTY_DIR}/boost-outcome/include)

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

    set(TEST_SOURCE_FILES
	${TEST_DIR}/main.cpp
	${TEST_DIR}/test_policy.cpp
	${TEST_DIR}/memory_leak_detector.cpp
	${TEST_DIR}/global_context_test.cpp
	${TEST_DIR}/thread_context_test.cpp
	${TEST_DIR}/cuda_exception_test.cpp
	${TEST_DIR}/cuda_error_test.cpp
	${TEST_DIR}/cuda_memory_test.cpp
	${TEST_DIR}/device_allocators_test.cpp
	${TEST_DIR}/mgblas_helpers_test.cpp
	${TEST_DIR}/device_matrix_test.cpp
	${TEST_DIR}/device_matrix_test.cpp
	${TEST_DIR}/matrix_view_test.cpp
	${TEST_DIR}/device_vector_test.cpp
	#${TEST_DIR}/type_trait_test.cpp
	${TEST_DIR}/blaslv1_operation_test.cpp
	${TEST_DIR}/blaslv3_operation_test.cpp
	${TEST_DIR}/blaslv1_expression_test.cpp
	${TEST_DIR}/blaslv3_expression_test.cpp
        ${TEST_DIR}/blaslv2_expression_test.cpp
	${TEST_DIR}/fft_test.cpp)
    mgcpp_display_elements("Source files for ${PROJECT_NAME} tests"
	"${TEST_SOURCE_FILES}")

    # add test target
    add_executable(${PROJECT_NAME}_test ${TEST_SOURCE_FILES})

    set(MGCPP_TARGETS ${MGCPP_TARGETS}
	"${PROJECT_NAME}_test")

    target_include_directories(${PROJECT_NAME}_test
	PUBLIC
	${PROJECT_SOURCE_DIR}/include
	${GTEST_INCLUDE_DIRS})

    target_link_libraries(${PROJECT_NAME}_test
	${PROJECT_NAME}
	${GTEST_LIBRARIES})

    set(TEST_BUILD_FLAGS ${BUILD_FLAGS})

    message("Test C++ flags = ${TEST_BUILD_FLAGS}")

    set_target_properties(${PROJECT_NAME}_test
	PROPERTIES COMPILE_FLAGS "${TEST_BUILD_FLAGS}")

    # ctest
    add_test(${PROJECT_NAME}_cmake_test ${PROJECT_NAME}_test)

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
