cmake_minimum_required(VERSION 3.15...3.26)

project(
  snail
  VERSION ${SKBUILD_PROJECT_VERSION}
  # C as well as C++: the vendored nanoarrow amalgamation is C99
  LANGUAGES C CXX)


set(CPP_SRC_DIR extension/src)
# Vendored rather than fetched, so that building from an sdist needs no
# network. Regenerate with scripts/vendor_nanoarrow.py; see the README there
# for why its symbols are namespaced.
set(NANOARROW_DIR extension/vendor/nanoarrow)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(intersections MODULE ${CPP_SRC_DIR}/intersections.cpp WITH_SOABI
                   ${CPP_SRC_DIR}/geoarrow.cpp ${CPP_SRC_DIR}/geometry.cpp
                   ${CPP_SRC_DIR}/operations.cpp
                   ${NANOARROW_DIR}/nanoarrow.c)

target_compile_features(intersections PRIVATE cxx_std_17)
set_target_properties(intersections PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
target_include_directories(intersections PUBLIC ${CPP_SRC_DIR})
# SYSTEM: vendored code is not ours to keep warning-free
target_include_directories(intersections SYSTEM PRIVATE ${NANOARROW_DIR})
target_link_libraries(intersections PRIVATE pybind11::headers)
target_compile_definitions(intersections PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS intersections DESTINATION snail/core)
