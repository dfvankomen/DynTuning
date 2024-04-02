cmake_minimum_required(VERSION 3.22)
project(DynamicTuningPrototype VERSION 1.0 LANGUAGES CXX)

# options to set execution spaces
option(Kokkos_DEVICE "Execution space for device code" OFF)

# Requires KOKKOS
find_package(Kokkos REQUIRED HINTS "./kokkos-4.1.00/build")

# set the host execution space macro
list(FIND Kokkos_DEVICES "SERIAL" _idx_SERIAL)
list(FIND Kokkos_DEVICES "OPENMP" _idx_OPENMP)
if(${_idx_SERIAL} GREATER -1)
  set(Kokkos_HOST "Serial")
elseif(${_idx_OPENMP} GREATER -1)
  set(Kokkos_HOST "OpenMP")
else()
  message(FATAL_ERROR "Kokkos was not compiled with a Host execution space!")
endif()
message(STATUS "Kokkos Host Execution Space: Kokkos::" ${Kokkos_HOST})

# set the device execution space macro
if(Kokkos_DEVICE)
  list(FIND Kokkos_DEVICES "CUDA" _idx_CUDA)
  list(FIND Kokkos_DEVICES "HIP"  _idx_HIP)
  list(FIND Kokkos_DEVICES "SYCL" _idx_SYCL)
  if(${_idx_CUDA} GREATER -1)
    set(Kokkos_DEVICE "Cuda")
    set(Kokkos_ENABLE_CUDA_LAMBDA ON)
  elseif(${_idx_HIP} GREATER -1)
    set(Kokkos_DEVICE "HIP")
  elseif(${_idx_SYCL} GREATER -1)
    set(Kokkos_DEVICE "Experimental::SYCL")
  else()
    set(Kokkos_DEVICE ${Kokkos_HOST})
    message(WARNING "Kokkos was not compiled with a Device execution space, defaulting to Host")
  endif()
else()
  set(Kokkos_DEVICE ${Kokkos_HOST})
  message(WARNING "Kokkos Device disabled, defaulting to Host")
endif()
message(STATUS "Kokkos Device Execution Space: Kokkos::" ${Kokkos_DEVICE})

# Build
add_executable(exe src/main.cpp)
target_link_libraries(exe PRIVATE Kokkos::kokkos)
#target_include_directories(exe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/eigen/Eigen")
target_compile_options(exe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --extended-lambdas
                       >)
target_compile_features(exe PRIVATE cxx_std_20)
target_compile_definitions(exe PUBLIC
  KOKKOS_HOST=${Kokkos_HOST}
  KOKKOS_DEVICE=${Kokkos_DEVICE}
)
set_target_properties(exe PROPERTIES OUTPUT_NAME ${PROJECT_NAME})


