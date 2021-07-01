add_subdirectory(lib/Catch2)
add_executable(tests tests.cpp ../find_intersections_linestring.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2)
