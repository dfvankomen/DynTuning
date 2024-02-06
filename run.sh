#!/bin/bash

inst="${cntr_inst:-kokkos}"
exe="${exe_name:?no executable specified}"

# run the executable
cd build && singularity exec "instance://${inst}" "./${exe}" $@
