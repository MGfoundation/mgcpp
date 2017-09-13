cmake_minimum_required(VERSION 3.8)
project(mgcpp)

# options
option(USE_MAGMA "compile mgcpp with magma features" TRUE)

# cmake modules path
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)


# cuda dependency
find_package(CUDA REQUIRED)

# set(CUDA_NVCC_FLAGS
#     # "-gencode arch=compute_30, code=sm_30;
#     "${CUDA_NVCC_FLAGS} --compiler-options ${BUILD_FLAGS}")

list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30; -O3")

message("build flags = ${CUDA_NVCC_FLAGS}")

# project source files
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(SOURCE_FILES
    # ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/initialize.c)

message("[ Source files for ${PROJECT_NAME} ]")
foreach(SOURCE_FILE ${SOURCE_FILES})
    message(STATUS "${SOURCE_FILE}")
endforeach()

cuda_add_library(${PROJECT_NAME} STATIC SHARED ${SOURCE_FILES})

## cmgcpp dependencies
# cuBLAS
cuda_add_cublas_to_target(${PROJECT_NAME})


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

