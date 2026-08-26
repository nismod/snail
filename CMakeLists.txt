cmake_minimum_required(VERSION 3.15...3.26)

project(
  snail
  VERSION ${SKBUILD_PROJECT_VERSION}
  # C as well as C++: the vendored nanoarrow amalgamation is C99
  LANGUAGES C CXX)


set(CPP_SRC_DIR extension/src)
# Vendored rather than fetched, so that building from an sdist needs no
# network. Regenerate with scripts/vendor_nanoarrow.py and
# scripts/vendor_geoarrow.py; see those for why the symbols are namespaced.
#
# Both amalgamations include each other by directory - geoarrow.c asks for
# "nanoarrow/nanoarrow.h" - so extension/vendor is the include path they share,
# and NANOARROW_DIR is on the path as well for our own flat "nanoarrow.hpp".
set(VENDOR_DIR extension/vendor)
set(NANOARROW_DIR ${VENDOR_DIR}/nanoarrow)
set(GEOARROW_DIR ${VENDOR_DIR}/geoarrow)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# The warnings we hold our own code to. Not MSVC - the flags are GCC/Clang
# syntax - and not the vendored C below: -isystem only quiets warnings from
# a header reached through it, not from a .c file compiled directly out of
# that directory, so the vendored sources are kept in a target of their own
# that this is never applied to.
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
