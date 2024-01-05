cmake_minimum_required(VERSION 3.23)
project(
  DynamicTuningPrototype
  VERSION 1.0
  LANGUAGES CXX)

add_subdirectory(kokkos-4.1.00)

add_executable(DynamicTuningPrototype main.cpp)
target_link_libraries(DynamicTuningPrototype PRIVATE Kokkos::kokkos)
