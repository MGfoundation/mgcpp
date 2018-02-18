# example source files

find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/examples/MLP)

set(EXAMPLE_SOURCE_FILES
    ${EXAMPLE_DIR}/MLP.cpp
    ${EXAMPLE_DIR}/mainwindow.cpp
    ${EXAMPLE_DIR}/drawwidget.cpp)

set(EXAMPLE_FORMS
    ${EXAMPLE_DIR}/mainwindow.ui)
qt5_wrap_ui(EXAMPLE_FORMS_H ${EXAMPLE_FORMS})

# add example target
add_executable(${PROJECT_NAME}_example_MLP ${EXAMPLE_SOURCE_FILES} ${EXAMPLE_FORMS_H})

set(MGCPP_TARGETS ${MGCPP_TARGETS}
    "${PROJECT_NAME}_example_MLP")

target_include_directories(${PROJECT_NAME}_example_MLP
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    )

target_link_libraries(${PROJECT_NAME}_example_MLP
    ${PROJECT_NAME}
    Qt5::Widgets
    )

set(EXAMPLE_BUILD_FLAGS ${BUILD_FLAGS})

set_target_properties(${PROJECT_NAME}_example_MLP
    PROPERTIES COMPILE_FLAGS "${EXAMPLE_BUILD_FLAGS}")
