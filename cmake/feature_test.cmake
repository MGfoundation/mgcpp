
set(FEATURE_TEST_DIR ${CMAKE_CURRENT_LIST_DIR}/feature_test)

message(STATUS "compiling feature test")
try_compile(BUILD_RESULT
    ${FEATURE_TEST_DIR}
    ${FEATURE_TEST_DIR}
    "feature_test"
    OUTPUT_VARIABLE BUILD_STATUS)

if(NOT BUILD_RESULT)
    message(${BUILD_STATUS})
    message(FATAL_ERROR "failed to build feature test")
endif()
message(STATUS "successfully compiled feature test")
message(STATUS "running feature test")

execute_process(COMMAND "${FEATURE_TEST_DIR}/cuda_gencode"
    RESULT_VARIABLE CUDA_GENCODE_TEST_RESULT
    OUTPUT_VARIABLE CUDA_GENCODE_TEST_STATUS)

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

