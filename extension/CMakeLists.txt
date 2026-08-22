cmake_minimum_required(VERSION 3.15...3.26)
project(snail)

set(CPP_SRC_DIR src)
set(CPP_TESTS_DIR tests)

add_subdirectory(extern/Catch2)
add_executable(run_tests
  ${CPP_TESTS_DIR}/run_tests.cpp
  ${CPP_TESTS_DIR}/tests_intersections.cpp
  ${CPP_TESTS_DIR}/tests_stress.cpp
  ${CPP_TESTS_DIR}/tests_transform.cpp
  ${CPP_SRC_DIR}/operations.cpp
)
target_include_directories(run_tests PUBLIC ${CPP_SRC_DIR})
target_link_libraries(run_tests PRIVATE Catch2::Catch2)

add_executable(run_benchmarks
  benchmarks/benchmark_split.cpp
  ${CPP_SRC_DIR}/operations.cpp
)
target_include_directories(run_benchmarks PUBLIC ${CPP_SRC_DIR})
if(NOT MSVC)
  target_compile_options(run_benchmarks PRIVATE -O2)
endif()
