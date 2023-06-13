cmake_minimum_required(VERSION 3.15...3.26)

project(
  snail
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)


set(CPP_SRC_DIR extension/src)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(intersections MODULE ${CPP_SRC_DIR}/intersections.cpp WITH_SOABI ${CPP_SRC_DIR}/operations.cpp)

target_include_directories(intersections PUBLIC ${CPP_SRC_DIR})
target_link_libraries(intersections PRIVATE pybind11::headers)
target_compile_definitions(intersections PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS intersections DESTINATION snail/core)
