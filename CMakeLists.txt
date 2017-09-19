cmake_minimum_required(VERSION 3.8)
project(mgcpp)

# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ${CMAKE_SOURCE_DIR}/lib)

# options
option(USE_MAGMA "compile mgcpp with magma features" TRUE)
option(USE_CUBLAS "compile mgcpp with cublas features" TRUE)


# cmake modules path
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)


# cuda dependency
find_package(CUDA REQUIRED)

list(APPEND CUDA_NVCC_FLAGS "-arch=sm_52; code=sm_52;-O3")

message("build flags = ${CUDA_NVCC_FLAGS}")


# project source files
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

set(BASE_SOURCE_FILES
    ${SOURCE_DIR}/cuda_stdlib.cpp
    ${SOURCE_DIR}/cuda_error.cpp)

set(MAIN_SOURCE_FILES "")
#${SOURCE_DIR}/)

message("[ Source files for ${PROJECT_NAME}base ]")
foreach(BASE_SOURCE_FILE ${BASE_SOURCE_FILES})
    message(STATUS "${BASE_SOURCE_FILE}")
endforeach()

message("[ Source files for ${PROJECT_NAME} ]")
foreach(SOURCE_FILE ${MAIN_SOURCE_FILES})
    message(STATUS "${SOURCE_FILE}")
endforeach()

cuda_add_library("${PROJECT_NAME}base" ${BASE_SOURCE_FILES} STATIC)

add_library(${PROJECT_NAME} STATIC SHARED ${MAIN_SOURCE_FILES})

set(BUILD_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -fPIC -Wall -Wextra")
message("C++ flags = ${BUILD_FLAGS}")

set_target_properties(${PROJECT_NAME}
    PROPERTIES COMPILE_FLAGS "${BUILD_FLAGS}")


# Include directories
target_include_directories("${PROJECT_NAME}base"
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

target_include_directories("${PROJECT_NAME}"
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include)


## mgcpp dependencies
# cuBLAS
if(USE_CUBLAS)
    cuda_add_cublas_to_target("${PROJECT_NAME}")
endif()


# mgcpp base
target_link_libraries(${PROJECT_NAME}
    "${PROJECT_NAME}base")


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


# Tests
set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)

set(TEST_SOURCE_FILES
    ${TEST_DIR}/main.cpp
    ${TEST_DIR}/cuda_stdlib_test.cpp
    ${TEST_DIR}/cuda_error_test.cpp)

message("[ Source files for ${PROJECT_NAME} tests ]")
foreach(TEST_SOURCE_FILE ${TEST_SOURCE_FILES})
    message(STATUS "${TEST_SOURCE_FILE}")
endforeach()

add_executable(${PROJECT_NAME}_test ${TEST_SOURCE_FILES})

target_include_directories(${PROJECT_NAME}_test
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${TEST_DIR}/Catch/include)

target_link_libraries(${PROJECT_NAME}_test
    ${PROJECT_NAME})

set_target_properties(${PROJECT_NAME}_test
    PROPERTIES COMPILE_FLAGS "${BUILD_FLAGS}")


# install
install(TARGETS "${PROJECT_NAME}base" ${PROJECT_NAME} "${PROJECT_NAME}_test"
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
    ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib
    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib)

