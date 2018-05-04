
## All these files will be deprecated!
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

# System Test
set(SYSTEM_TEST_PATH ${TEST_DIR}/system)

# Allocator Test
set(ALLOCATOR_TEST_PATH ${TEST_DIR}/allocators)

# Adapters Test
set(ADAPTERS_TEST_PATH ${TEST_DIR}/adapters)

# Context Test
set(CONTEXT_TEST_PATH ${TEST_DIR}/context)

# CUDA Test
set(CUDA_TEST_PATH ${TEST_DIR}/cuda)

# CUDA LIBS Test
set(CUDA_LIBS_TEST_PATH ${TEST_DIR}/cuda_libs)

# Matrix Test
set(MATRIX_TEST_PATH ${TEST_DIR}/matrix)
set(TEST_SOURCE_FILES
    ${TEST_SOURCE_FILES}
    ${MATRIX_TEST_PATH}/device_matrix_test.cpp)

# Operations Test
set(OPERATIONS_TEST_PATH ${TEST_DIR}/operations)

# Expressions Test
set(EXPRESSIONS_TEST_PATH ${TEST_DIR}/expressions)

# Vector Test
set(VECTOR_TEST_PATH ${TEST_DIR}/vector)

# Operations Test
set(OPERATIONS_TEST_PATH ${TEST_DIR}/operations)

# Kernels Test
set(KERNELS_TEST_PATH ${TEST_DIR}/kernels)

# Type Traits Test
set(TYPE_TRAITS_TEST_PATH ${TEST_DIR}/type_traits)

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
