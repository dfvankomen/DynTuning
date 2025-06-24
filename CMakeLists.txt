cmake_minimum_required(VERSION 3.22)
project(DynamicTuningPrototype VERSION 1.0 LANGUAGES CXX)

# options to set execution spaces
option(Kokkos_DEVICE "Execution space for device code" OFF)

# Requires KOKKOS
find_package(Kokkos REQUIRED HINTS "./kokkos-4.1.00/build")
find_package(KokkosKernels REQUIRED HINTS "./kokkos-kernels-4.1.00/build")

# set the host execution space macro
list(FIND Kokkos_DEVICES "SERIAL" _idx_SERIAL)
list(FIND Kokkos_DEVICES "OPENMP" _idx_OPENMP)
list(FIND Kokkos_DEVICES "THREADS" _idx_THREADS)
if(${_idx_SERIAL} GREATER -1)
  set(Kokkos_HOST "Serial")
elseif(${_idx_OPENMP} GREATER -1)
  set(Kokkos_HOST "OpenMP")
elseif(${_idx_THREADS} GREATER -1)
  set(Kokkos_HOST "Threads")
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

option(DYNTUNE_ENABLE_ORDER_SHUFFLE "Whether or not to shuffle the kernel operation order" ON)
option(DYNTUNE_SINGLE_CHAIN_RUN "Only allow a single chain to run, requires --single_chain option" OFF)
option(DYNTUNE_DEBUG_ENABLED "Enables a large number of debug statements" OFF)

if(DYNTUNE_ENABLE_ORDER_SHUFFLE)
  add_definitions(-DDYNTUNE_ENABLE_ORDER_SHUFFLE)
endif()

if(DYNTUNE_SINGLE_CHAIN_RUN)
  add_definitions(-DDYNTUNE_SINGLE_CHAIN_RUN)
endif()

if(DYNTUNE_DEBUG_ENABLED)
  add_definitions(-DDYNTUNE_DEBUG_ENABLED)
endif()

find_package(Eigen3 REQUIRED)
if (TARGET Eigen3::Eigen)
  message(STATUS "Eigen3 was found!")
else()
  message(FATAL_ERROR "Eigen3 was not found!")
endif()


# Build
add_executable(exe src/main.cpp)
target_link_libraries(exe PRIVATE Kokkos::kokkos)
target_link_libraries(exe PRIVATE Eigen3::Eigen)
target_include_directories(exe PRIVATE
 "${CUDA_HOME}/include"
  src
  src/kernels
)
add_definitions(-w)
target_compile_options(exe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambdas
  >
)
target_compile_features(exe PRIVATE cxx_std_20)
target_compile_definitions(exe PUBLIC
  KOKKOS_HOST=${Kokkos_HOST}
  KOKKOS_DEVICE=${Kokkos_DEVICE}
)
set_target_properties(exe PROPERTIES OUTPUT_NAME ${PROJECT_NAME})




# test build
add_executable(kernelBlocks src/test_kernel_block_sizes.cpp)
target_link_libraries(kernelBlocks PRIVATE Kokkos::kokkos)
target_link_libraries(kernelBlocks PRIVATE Eigen3::Eigen)
target_include_directories(kernelBlocks PRIVATE
 "${CUDA_HOME}/include"
  src
  src/kernels
)
add_definitions(-w)
target_compile_options(kernelBlocks PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambdas
  >
)
target_compile_features(kernelBlocks PRIVATE cxx_std_20)
target_compile_definitions(kernelBlocks PUBLIC
  KOKKOS_HOST=${Kokkos_HOST}
  KOKKOS_DEVICE=${Kokkos_DEVICE}
)


# test for threads
add_executable(kernelThreads src/test_kernel_thread_sizes.cpp)
target_link_libraries(kernelThreads PRIVATE Kokkos::kokkos)
target_link_libraries(kernelThreads PRIVATE Eigen3::Eigen)
target_include_directories(kernelThreads PRIVATE
 "${CMAKE_CURRENT_SOURCE_DIR}/eigen/Eigen"
 "${CUDA_HOME}/include"
  src
  src/kernels
)
add_definitions(-w)
target_compile_options(kernelThreads PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambdas
  >
)
target_compile_features(kernelThreads PRIVATE cxx_std_20)
target_compile_definitions(kernelThreads PUBLIC
  KOKKOS_HOST=${Kokkos_HOST}
  KOKKOS_DEVICE=${Kokkos_DEVICE}
)



## TEST CASES
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)

add_executable(tests tests/main.cpp tests/data.test.cpp tests/kernels.test.cpp tests/optimizer.test.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(tests PRIVATE Kokkos::kokkos)
target_link_libraries(tests PRIVATE Kokkos::kokkoskernels)
target_link_libraries(tests PRIVATE Eigen3::Eigen)
target_include_directories(tests PRIVATE
 "${CUDA_HOME}/include"
  src
  src/kernels
)
add_definitions(-w)
target_compile_options(tests PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambdas
  >
)
target_compile_features(tests PRIVATE cxx_std_20)
target_compile_definitions(tests PUBLIC
  KOKKOS_HOST=${Kokkos_HOST}
  KOKKOS_DEVICE=${Kokkos_DEVICE}
)

# add catch 2 to the cmake module path, then include CTest and Catch
# NOTE: this will fail on the docker build, so ENABLE_RUN_TESTS is set to off by default
if(ENABLE_RUN_TESTS)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(CTest)
  include(Catch)
  catch_discover_tests(tests)
endif()
