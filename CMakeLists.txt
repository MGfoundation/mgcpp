cmake_minimum_required(VERSION 3.8)
project(mgcpp)

# options
option(USE_MAGMA "compile mgcpp with magma features" TRUE)

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
    ${SOURCE_DIR}/cuda_stdlib.cpp)

set(MAIN_SOURCE_FILES
    ${SOURCE_DIR}/main.cpp)

message("[ Source files for ${PROJECT_NAME}base ]")
foreach(BASE_SOURCE_FILE ${BASE_SOURCE_FILES})
    message(STATUS "${BASE_SOURCE_FILE}")
endforeach()

message("[ Source files for ${PROJECT_NAME} ]")
foreach(SOURCE_FILE ${MAIN_SOURCE_FILES})
    message(STATUS "${SOURCE_FILE}")
endforeach()

cuda_add_library("${PROJECT_NAME}base" ${BASE_SOURCE_FILES} STATIC)

add_executable(${PROJECT_NAME} # STATIC SHARED 
    ${MAIN_SOURCE_FILES})

add_dependencies(${PROJECT_NAME} "${PROJECT_NAME}base")

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 17)

# Include directories
target_include_directories("${PROJECT_NAME}base"
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

target_include_directories("${PROJECT_NAME}"
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include)


## mgcpp dependencies
# cuBLAS
cuda_add_cublas_to_target("${PROJECT_NAME}")


# mgcpp base
# target_link_libraries(${PROJECT_NAME}
#     "${PROJECT_SOURCE_DIR}/lib/libmgcppbase.a")
# target_link_libraries(${PROJECT_NAME}
#     "${PROJECT_SOURCE_DIR}/lib/libmgcppbase.so")
target_link_libraries(${PROJECT_NAME}
    "${PROJECT_SOURCE_DIR}/lib${PROJECT_NAME}base.so")


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


# install
install(TARGETS "${PROJECT_NAME}base" ${PROJECT_NAME}
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
    ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib
    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib)

