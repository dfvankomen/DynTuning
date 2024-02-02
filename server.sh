#!/bin/bash

cntr="${cntr_name:-cntr/kokkos.sif}"
inst="${cntr_inst:-kokkos}"

# check if an instance by this name is already running
stat=0
for name in $(IFS=$'\n'; for line in $(singularity instance list | tail -n +2); do echo $line | xargs | cut -d ' ' -f 1; done); do
  if [[ "${name}" == "${inst}" ]]; then
    stat=1
  fi
done

# start or stop a container
if [[ "${stat}" == 0 ]]; then
  singularity instance start "${cntr}" "${inst}"
elif [[ "${stat}" == 1 ]]; then
  singularity instance stop "${inst}"
fi
 
