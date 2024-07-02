#!/bin/bash

# Helper script to automate calling the three "main" build scripts
# I got tired of having to remember which three are necessary
# for Google cloud. Google cloud doesn't need the kokkos container
# nor the server for a development enviornment. The commands
# in the README script are perfectly fine for if there's a
# container that needs to be built and run.

TEMP_CURR_DIR=$(pwd)

# set up the environment
. ./setup-env.sh googlecloud

# then prepare the build environment
. ./install-env.sh

cd "${TEMP_CURR_DIR}"
unset TEMP_CURR_DIR

# then set up the build
. ./build.sh
