#!/bin/bash

name="${cntr_name:-kokkos}"

[[ -f "${name}.sif" ]] || singularity build -f "${name}.sif" "${name}.def"
