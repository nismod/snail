cmake_minimum_required(VERSION 3.15...3.26)
# C as well as C++: the vendored amalgamations are C99
project(snail LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CPP_SRC_DIR src)
set(CPP_TESTS_DIR tests)
# The Arrow layer is built here too. It has no pybind11 in it - that is the
# whole point of it being a file of its own - so the Catch2 suite can reach
# the reading, splitting and writing of GeoArrow directly, which it could
# not while that code sat beside the bindings.
set(VENDOR_DIR vendor)

add_subdirectory(extern/Catch2)
add_executable(run_tests
  ${CPP_TESTS_DIR}/run_tests.cpp
  ${CPP_TESTS_DIR}/tests_intersections.cpp
  ${CPP_TESTS_DIR}/tests_geoarrow.cpp
  ${CPP_TESTS_DIR}/tests_stress.cpp
  ${CPP_TESTS_DIR}/tests_transform.cpp
  ${CPP_SRC_DIR}/geoarrow_arrays.cpp
  ${CPP_SRC_DIR}/geometry.cpp
  ${CPP_SRC_DIR}/operations.cpp
  ${VENDOR_DIR}/nanoarrow/nanoarrow.c
  ${VENDOR_DIR}/geoarrow/geoarrow.c
)
set_target_properties(run_tests PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
target_include_directories(run_tests PUBLIC ${CPP_SRC_DIR})
# SYSTEM: vendored code is not ours to keep warning-free
target_include_directories(run_tests SYSTEM PRIVATE
  ${VENDOR_DIR} ${VENDOR_DIR}/nanoarrow)
target_link_libraries(run_tests PRIVATE Catch2::Catch2)

add_executable(run_benchmarks
  benchmarks/benchmark_split.cpp
  ${CPP_SRC_DIR}/operations.cpp
)
target_include_directories(run_benchmarks PUBLIC ${CPP_SRC_DIR})
if(NOT MSVC)
  target_compile_options(run_benchmarks PRIVATE -O2)
endif()
