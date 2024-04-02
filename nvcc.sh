#!/bin/bash
singularity exec instance://kokkos nvcc_wrapper $@
