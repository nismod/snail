cmake_minimum_required(VERSION 3.10)
project(snail)

set(CPP_SRC_DIR src/cpp)
set(CPP_TESTS_DIR tests/cpp)

# "SKBUILD" is true if cmake is called from scikit-build, typically
# building the wheel from pip. If this is the case, we don't build the
# test executable.
if (NOT SKBUILD)
  add_subdirectory(${CPP_TESTS_DIR}/lib/Catch2)
  add_executable(run_tests
    ${CPP_TESTS_DIR}/run_tests.cpp
    ${CPP_TESTS_DIR}/tests_intersections.cpp
    ${CPP_TESTS_DIR}/tests_transform.cpp
    ${CPP_SRC_DIR}/operations.cpp
  )
  target_include_directories(run_tests PUBLIC ${CPP_SRC_DIR})
  target_link_libraries(run_tests PRIVATE Catch2::Catch2)

else()
  add_subdirectory(pybind11)
  pybind11_add_module(intersections
    ${CPP_SRC_DIR}/intersections.cpp
    ${CPP_SRC_DIR}/operations.cpp
  )
  install(TARGETS intersections LIBRARY DESTINATION .)

endif()
