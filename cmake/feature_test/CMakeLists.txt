cmake_minimum_required(VERSION 3.8)
project(feature_test)

find_package(CUDA REQUIRED)
add_executable(cuda_gencode "${PROJECT_SOURCE_DIR}/cuda_gencode_test.cpp")
target_include_directories(cuda_gencode
    PUBLIC
    ${CUDA_INCLUDE_DIRS})

target_link_libraries(cuda_gencode
    ${CUDA_LIBRARIES})
