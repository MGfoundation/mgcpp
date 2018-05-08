
set(FEATURE_TEST_DIR ${CMAKE_CURRENT_LIST_DIR}/feature_test)

message(STATUS "compiling and running feature test")

find_package(CUDA REQUIRED)
try_run(CUDA_GENCODE_TEST_RESULT BUILD_RESULT
    "${CMAKE_CURRENT_BINARY_DIR}/FeatureTest"
    SOURCES "${FEATURE_TEST_DIR}/cuda_gencode_test.cpp"
	LINK_LIBRARIES "${CUDA_LIBRARIES}"
	CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
				"-DLINK_DIRECTORIES=${CUDA_LIBRARIES}"
    COMPILE_OUTPUT_VARIABLE BUILD_STATUS
	RUN_OUTPUT_VARIABLE CUDA_GENCODE_TEST_STATUS)

if(NOT BUILD_RESULT)
    message(${BUILD_STATUS})
    message(FATAL_ERROR "failed to build feature test")
endif()
message(STATUS "successfully compiled feature test")

if(NOT ${CUDA_GENCODE_TEST_RESULT} EQUAL "0")
    message(STATUS "running feature test - failed")
    message(STATUS "disabling custom kernel build")
    message("${CUDA_GENCODE_TEST_STATUS}")
    set(BUILD_CUSTOM_KERNELS OFF)
else()
    string(REGEX REPLACE "\n$" ""
	CUDA_GENCODE_TEST_STATUS "${CUDA_GENCODE_TEST_STATUS}")

    message(STATUS "running feature test - success")
    message(STATUS "setting NVCC flags ${CUDA_GENCODE_TEST_STATUS}")
    set(CUDA_GEN_CODE "${CUDA_GENCODE_TEST_STATUS}")
endif()

