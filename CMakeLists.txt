cmake_minimum_required(VERSION 3.8)
project(mgcpp)

# cmake modules path
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# cuda dependency
find_package(CUDA REQUIRED)

# project source files
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(SOURCE_FILES ${SOURCE_DIR}/main.cpp)

message("[ Source files for ${PROJECT_NAME} ]")
foreach(SOURCE_FILE ${SOURCE_FILES})
    message(STATUS "${SOURCE_FILE}")
endforeach()

cuda_add_executable(${PROJECT_NAME} ${SOURCE_FILES})

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
    CXX_STANDARD 17)

# Include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include)


# cuda includes
target_include_directories(${PROJECT_NAME})


## Thread library dependency
find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_THREAD_LIBS_INIT})

# install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin)

