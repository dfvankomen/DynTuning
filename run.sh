#!/bin/bash

# USAGE
# run this script with an argument which specifies the target device
# e.g.
# ./run.sh host
# or
# ./run.sh device

exe="${exe_name:?no executable specified}"

# run the executable
cd build
if [[ -n "${cntr_inst}" ]]; then
  singularity exec "instance://${cntr_inst}" "./${exe}" $@
else
  "./${exe}" $@
fi
