cmake_minimum_required(VERSION 3.15...3.26)

project(
  snail
  VERSION ${SKBUILD_PROJECT_VERSION}
  # C as well as C++: the vendored nanoarrow amalgamation is C99
  LANGUAGES C CXX)


set(CPP_SRC_DIR extension/src)
set(VENDOR_DIR extension/extern)
set(NANOARROW_DIR ${VENDOR_DIR}/nanoarrow)
set(GEOARROW_DIR ${VENDOR_DIR}/geoarrow)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Warnings for snail (GCC/Clang, not MSVC; not vendored code)
if(NOT MSVC)
  set(SNAIL_WARNING_FLAGS -Wall -Wextra -Wpedantic)
endif()

add_library(vendor_arrow OBJECT ${NANOARROW_DIR}/nanoarrow.c ${GEOARROW_DIR}/geoarrow.c)
set_target_properties(vendor_arrow PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
target_include_directories(vendor_arrow PRIVATE ${VENDOR_DIR} ${NANOARROW_DIR})

python_add_library(intersections MODULE ${CPP_SRC_DIR}/intersections.cpp WITH_SOABI
                   ${CPP_SRC_DIR}/geoarrow.cpp ${CPP_SRC_DIR}/geoarrow_arrays.cpp
                   ${CPP_SRC_DIR}/geometry.cpp
                   ${CPP_SRC_DIR}/operations.cpp
                   $<TARGET_OBJECTS:vendor_arrow>)

target_compile_features(intersections PRIVATE cxx_std_17)
target_compile_options(intersections PRIVATE ${SNAIL_WARNING_FLAGS})
target_include_directories(intersections PUBLIC ${CPP_SRC_DIR})
# SYSTEM: vendored code is not ours to keep warning-free
target_include_directories(intersections SYSTEM PRIVATE ${VENDOR_DIR} ${NANOARROW_DIR})
target_link_libraries(intersections PRIVATE pybind11::headers)
target_compile_definitions(intersections PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS intersections DESTINATION snail/core)
