# example source files
set(EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/examples/MLP)

set(EXAMPLE_SOURCE_FILES
    ${EXAMPLE_DIR}/MLP.cpp)
mgcpp_display_elements("Source files for ${PROJECT_NAME} MLP example"
    "${EXAMPLE_SOURCE_FILES}")

# add example target
add_executable(${PROJECT_NAME}_example_MLP ${EXAMPLE_SOURCE_FILES})

set(MGCPP_TARGETS ${MGCPP_TARGETS}
    "${PROJECT_NAME}_example_MLP")

target_include_directories(${PROJECT_NAME}_example_MLP
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME}_example_MLP
    ${PROJECT_NAME})

set(EXAMPLE_BUILD_FLAGS ${BUILD_FLAGS})

set_target_properties(${PROJECT_NAME}_example_MLP
    PROPERTIES COMPILE_FLAGS "${EXAMPLE_BUILD_FLAGS}")
