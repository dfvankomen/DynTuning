#!/bin/bash

inst="${cntr_inst:-kokkos}"

# always clean build for testing
rm -rf build
mkdir build
cd build

# run cmake
singularity exec "instance://${inst}" \
  cmake ../ \
    -DKokkos_ROOT=/usr/local/kokkos/lib/cmake/Kokkos \
    -DCMAKE_CXX_COMPILER=/usr/local/kokkos/bin/nvcc_wrapper \
    -DKokkos_DEVICE=ON

# run make
singularity exec "instance://${inst}" make

