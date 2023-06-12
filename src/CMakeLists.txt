cmake_minimum_required(VERSION 3.15...3.26)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(intersections MODULE extension/src/intersections.cpp WITH_SOABI)
target_link_libraries(intersections PRIVATE pybind11::headers)
target_compile_definitions(intersections PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS intersections DESTINATION snail/core)
