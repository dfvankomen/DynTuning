#!/bin/bash

# always clean build for testing
rm -rf build
mkdir build
cd build

cmake_cmd="cmake ../ \
  -DKokkos_ROOT=${KOKKOS_HOME}/lib/cmake/Kokkos \
  -DCMAKE_CXX_COMPILER=${KOKKOS_HOME}/bin/nvcc_wrapper \
  -DKokkos_DEVICE=ON"

make_cmd="make -j 8"

# run cmake
if [[ -n "${cntr_inst}" ]]; then
  singularity exec "instance://${cntr_inst}" ${cmake_cmd[@]}
  singularity exec "instance://${cntr_inst}" ${make_cmd[@]}
else
  ${cmake_cmd[@]}
  ${make_cmd[@]}
fi

