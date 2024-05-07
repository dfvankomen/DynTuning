#!/bin/bash

# USAGE
# source this script with an argument for which environment to initialize
# e.g.
# . ./setup-env.sh narwhal
# or
# . ./setup-env.sh googlecloud

error=0
if [[ "${1}" == "narwhal" ]]; then

  # set home paths
  export CUDA_HOME=/usr/local/cuda
  export KOKKOS_HOME=/usr/local/kokkos

  # set the container path and instance name
  export cntr_name=cntr/kokkos-420-gcc-114-cuda-120.sif
  export cntr_inst=kokkos

  # load singularity or container support
  module load singularity/3.8.4

  # always bind the tmpdir
  export TMPDIR=/p/work1/tmp/$(whoami)
  export SINGULARITY_BIND="${TMPDIR},${SINGULARITY_BIND}"

  # always bind this directory
  export SINGULARITY_BIND="$(pwd),${SINGULARITY_BIND}"

  # enable nvidia support
  export SINGULARITY_NV=1
  
  # set the Kokkos_ARCH values
  export CPU_ARCH=ZEN2
  export GPU_ARCH=VOLTA70

  # set number of processors to use for compiling
  export MAKE_NPROCS=32

elif [[ "${1}" == "googlecloud" ]]; then
  
  # set home paths
  export CUDA_HOME=/usr/local/cuda
  export KOKKOS_HOME=~/opt/kokkos

  # ensure CUDA is on the path
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
  
  # ensure KOKKOS is on the path
  export PATH="${KOKKOS_HOME}/bin:${PATH}"
  
  # set the Kokkos_ARCH values
  export CPU_ARCH=SKX
  export GPU_ARCH=TURING75

  # set number of processors to use for compiling
  export MAKE_NPROCS=8

else
  error=1
fi

# don't use exit in sourceable scripts
if [[ "${error}" == 0 ]]; then
  
  # set project executable name
  export exe_name="DynamicTuningPrototype"

  # ensure compiler is gcc
  export CC=gcc
  export CXX=g++
  export NVCC_WRAPPER_DEFAULT_COMPILER=g++

fi

unset error
