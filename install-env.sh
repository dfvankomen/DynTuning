#!/bin/bash
  
# install dependencies
sudo apt-get install -y cmake

# install cuda
if [[ ! -d "${CUDA_HOME}" ]]; then
  cd ~
  mkdir ~/src
  cd src
  wget "https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run"
  chmod +x cuda_12.0.1_525.85.12_linux.run
  sudo ./cuda_12.0.1_525.85.12_linux.run --silent --toolkit --toolkitpath="${CUDA_HOME}"
fi

# install kokkos
if [[ ! -d "${KOKKOS_HOME}" ]]; then
  cd ~
  mkdir ~/opt
  mkdir ~/src
  cd ~/src
  git clone --depth 1 --branch 4.2.00 "https://github.com/kokkos/kokkos.git"
  cd kokkos
  rm -rf build
  mkdir build
  cd build
  cmake ../ \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_INSTALL_PREFIX="${KOKKOS_HOME}" \
    -DKokkos_ENABLE_SERIAL=ON \
    "-DKokkos_ARCH_${CPU_ARCH}=ON" \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
    "-DKokkos_ARCH_${GPU_ARCH}=ON" \
    -DKokkos_CUDA_DIR="${CUDA_HOME}"
  make -j "${MAKE_NPROCS}" install
fi
