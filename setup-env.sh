#!/bin/bash

# set the container path and instance name
cntr_name=cntr/kokkos-420-gcc-114-cuda-120.sif
cntr_inst=kokkos
exe_name="DynamicTuningPrototype"

# load singularity or container support
module load singularity/3.8.4

# always bind the tmpdir
export TMPDIR=/p/work1/tmp/$(whoami)
export SINGULARITY_BIND="${TMPDIR},${SINGULARITY_BIND}"

# always bind this directory
export SINGULARITY_BIND="$(dirname $0),${SINGULARITY_BIND}"

# enable nvidia support
export SINGULARITY_NV=1

# ensure compiler is gcc
export CC=gcc
export CXX=g++
export NVCC_WRAPPER_DEFAULT_COMPILER=g++

