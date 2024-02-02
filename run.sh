#!/bin/bash

inst="${cntr_inst:-kokkos}"

# run the executable
cd build && singularity exec "instance://${inst}" "./${exe}" $@
