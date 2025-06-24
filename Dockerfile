FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        make \
        wget \
        git \
        cmake \
        libeigen3-dev \
        gdb \
        cmake-curses-gui \
        && rm -rf /var/lib/apt/lists/*


# we know that cuda is at /usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda

# we also need to set where KOKKOS will exist
ENV KOKKOS_HOME=/usr/local/kokkos
ENV KOKKOS_KERNELS_HOME=/usr/local/kokkos-kernels

# update the paths, ensure CUDA and KOKKOS are on path
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

ENV PATH=$KOKKOS_HOME/bin:$PATH
ENV PATH=$KOKKOS_KERNELS_HOME/bin:$PATH

# set the Kokkos_ARCH values, 
ENV CPU_ARCH=ZEN2
ENV GPU_ARCH=AMPERE80

# and then the openmp options
ENV OMP_PLACES=threads
ENV OMP_PROC_BIND=spread

# number of processors for compiling
ENV MAKE_NPROCS=8

# and then set the exenames and compilers
ENV exe_name="DynamicTuningPrototype"
ENV CC=gcc
ENV CXX=g++
ENV NVCC_WRAPPER_DEFAULT_COMPILER=g++

# set up the work directory to "work"
WORKDIR /work

# then we prepare the build environment, based on install-env.sh
# NOTE: Cuda is already installed, thanks to the NVIDIA base package!


# install KOKKOS and Kernel, and CUDA_HOME, but that directory should exist
COPY install-env.sh /work/install-env.sh

RUN bash /work/install-env.sh && rm -rf /var/lib/apt/lists/*

COPY . /work

RUN bash /work/build.sh

WORKDIR /work/build

CMD ["./DynamicTuningPrototype"]
