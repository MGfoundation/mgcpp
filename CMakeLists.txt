
#              Copyright RedPortal 2017 - 2017.
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


# options
option(USE_MAGMA "compile mgcpp with magma features" ON)
option(USE_CUSPARSE "compile mgcpp with cusparse features" ON)
option(BUILD_TESTING "compile mgcpp with tests" ON)


# cmake modules path
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)


# cuda dependency
find_package(CUDA REQUIRED)

list(APPEND CUDA_NVCC_FLAGS "-arch=sm_61; code=sm_61; -O3")

message("build flags = ${CUDA_NVCC_FLAGS}")


# project source files
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(KERNELS_SOURCE_DIR ${SOURCE_DIR}/kernels)

set(BASE_SOURCE_FILES "")
# ${KERNELS_SOURCE_DIR}/fill.cu)

set(MAIN_SOURCE_FILES 
    ${SOURCE_DIR}/global_context.cpp
    ${SOURCE_DIR}/cublas_error.cpp
    ${SOURCE_DIR}/cuda_error.cpp
    ${SOURCE_DIR}/mgcpp_error_code.cpp
    ${SOURCE_DIR}/thread_context.cpp
    ${SOURCE_DIR}/leak_checker.cpp)

mgcpp_display_elements("Source files for ${PROJECT_NAME}base"
    "${BASE_SOURCE_FILES}")
mgcpp_display_elements("Source files for ${PROJECT_NAME}"
    "${MAIN_SOURCE_FILES}")

cuda_add_library("${PROJECT_NAME}base" STATIC ${BASE_SOURCE_FILES})

add_library(${PROJECT_NAME} STATIC SHARED ${MAIN_SOURCE_FILES})

if(MSVC)
    set(BUILD_FLAGS
	"${CMAKE_CXX_FLAGS} /w3 /Ot /Ox /MP")
else()
    set(BUILD_FLAGS
	"${CMAKE_CXX_FLAGS} -std=c++14 -g -fPIC -Wall -Wextra")
endif()
message("C++ flags = ${BUILD_FLAGS}")

set_target_properties(${PROJECT_NAME}
    PROPERTIES COMPILE_FLAGS "${BUILD_FLAGS}")


# Include directories
target_include_directories("${PROJECT_NAME}base"
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include)


## mgcpp dependencies
set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/third_party)

# cuBLAS
cuda_add_cublas_to_target("${PROJECT_NAME}")

# cuSPARSE
if(USE_CUSPARSE)
    target_link_libraries(${PROJECT_NAME}
	${CUDA_cusparse_LIBRARIES})
endif()

# add outcome library
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${THIRD_PARTY_DIR}/outcome/include)

# mgcpp base
target_link_libraries(${PROJECT_NAME}
    "${PROJECT_NAME}base")

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


# Tests
if(BUILD_TESTING)
    enable_testing()

    # find gtest dependency
    find_package(GTest REQUIRED)

    # test source files
    set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)

    set(TEST_SOURCE_FILES
	${TEST_DIR}/main.cpp
	${TEST_DIR}/cuda_stdlib_test.cpp
	${TEST_DIR}/gpu_matrix_test.cpp
	${TEST_DIR}/expression_mult_test.cpp
	# ${TEST_DIR}/tools_test.cpp
	${TEST_DIR}/cublas_helpers_test.cpp
	${TEST_DIR}/cpu_matrix_test.cpp
	${TEST_DIR}/cuda_exception_test.cpp
	${TEST_DIR}/cuda_error_test.cpp
	${TEST_DIR}/type_trait_test.cpp
	${TEST_DIR}/global_context_test.cpp
	${TEST_DIR}/thread_context_test.cpp
	${TEST_DIR}/operation_mult_test.cpp)
    mgcpp_display_elements("Source files for ${PROJECT_NAME} tests"
	${TEST_SOURCE_FILES})

    # add test target
    add_executable(${PROJECT_NAME}_test ${TEST_SOURCE_FILES})

    target_include_directories(${PROJECT_NAME}_test
	PUBLIC
	${PROJECT_SOURCE_DIR}/include
	${GTEST_INCLUDE_DIRS})

    target_link_libraries(${PROJECT_NAME}_test
	${PROJECT_NAME}
	${GTEST_LIBRARIES})

    set(TEST_BUILD_FLAGS
	"${CMAKE_CXX_FLAGS} -std=c++14 -g -Wall -Wextra")

    message("Test C++ flags = ${TEST_BUILD_FLAGS}")

    set_target_properties(${PROJECT_NAME}_test
	PROPERTIES COMPILE_FLAGS "${TEST_BUILD_FLAGS}")

    # ctest
    add_test(${PROJECT_NAME}_cmake_test ${PROJECT_NAME}_test)
endif()

# install
install(TARGETS
    "${PROJECT_NAME}base"
    "${PROJECT_NAME}"
    "${PROJECT_NAME}_test"
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
    ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib
    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib)
