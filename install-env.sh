#!/bin/bash

# install dependencies
sudo apt-get update
sudo apt-get install -y cmake gdb cmake-curses-gui

# install cuda
if [[ ! -d "${CUDA_HOME}" ]]; then
  echo "NOW INSTALLING CUDA TO $CUDA_HOME"
  cd ~
  # make the src and opt directores if they don't exist, otherwise cuda installer will under root
  mkdir -p ~/src
  mkdir -p ~/opt
  cd src
  # use the -nc command to avoid downloading the 4GB file if it already exists!
  wget -nc "https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run"
  chmod +x cuda_12.0.1_525.85.12_linux.run
  sudo ./cuda_12.0.1_525.85.12_linux.run --silent --toolkit --toolkitpath="${CUDA_HOME}"
fi

# install kokkos
if [[ ! -d "${KOKKOS_HOME}" ]]; then
  echo "NOW INSTALLING KOKKOS TO $KOKKOS_HOME"
  cd ~
  # -p asserts that they don't get overwritten
  mkdir -p ~/opt
  mkdir -p ~/src
  cd ~/src
  git clone --depth 1 --branch 4.2.00 "https://github.com/kokkos/kokkos.git"
  cd kokkos
  rm -rf build
  mkdir build
  cd build
  # cmake ../ \
  #   -DCMAKE_CXX_COMPILER="${CXX}" \
  #   -DCMAKE_INSTALL_PREFIX="${KOKKOS_HOME}" \
  #   -DKokkos_ENABLE_SERIAL=ON \
  #   "-DKokkos_ARCH_${CPU_ARCH}=ON" \
  #   -DKokkos_ENABLE_CUDA=ON \
  #   -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  #   -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
  #   "-DKokkos_ARCH_${GPU_ARCH}=ON" \
  #   -DKokkos_CUDA_DIR="${CUDA_HOME}"

  # OpenMP Version of the Install
  cmake ../ \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_INSTALL_PREFIX="${KOKKOS_HOME}" \
    -DKokkos_ENABLE_OPENMP=ON \
    "-DKokkos_ARCH_${CPU_ARCH}=ON" \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
    "-DKokkos_ARCH_${GPU_ARCH}=ON" \
    -DKokkos_CUDA_DIR="${CUDA_HOME}"

  # then run the make to actually build it!
  make -j "${MAKE_NPROCS}" install
fi

if [[ ! -d "${KOKKOS_KERNELS_HOME}" ]]; then
  echo "NOW INSTALLING KOKKOS_KERNELS TO $KOKKOS_KERNELS_HOME"
  cd ~
  mkdir -p ~/opt
  mkdir -p ~/src
  cd ~/src
  git clone --depth 1 --branch 4.2.00 "https://github.com/kokkos/kokkos-kernels.git"
  cd kokkos-kernels
  rm -rf build
  mkdir build
  cd build
  cmake ../ \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_INSTALL_PREFIX="${KOKKOS_KERNELS_HOME}" \
    -DKokkos_DIR="${KOKKOS_HOME}/lib/cmake/Kokkos" \
    -DKokkosKernels_DIR="${KOKKOS_KERNELS_HOME}/lib/cmake/KokkosKernels"
  make -j "${MAKE_NPROCS}" install
fi
