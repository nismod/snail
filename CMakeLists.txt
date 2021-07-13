cmake_minimum_required(VERSION 3.10)
project(snail)

set(CPP_SRC_DIR src/cpp)
set(CPP_TESTS_DIR tests/cpp)

add_subdirectory(${CPP_TESTS_DIR}/lib/Catch2)
add_executable(tests_intersections
  ${CPP_TESTS_DIR}/tests_intersections.cpp
  ${CPP_SRC_DIR}/find_intersections_linestring.cpp
  )
target_include_directories(tests_intersections PUBLIC ${CPP_SRC_DIR})
target_link_libraries(tests_intersections PRIVATE Catch2::Catch2)
