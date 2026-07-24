#!/bin/bash
# Builds and runs the gprat_distributed benchmark against the CPU (OpenBLAS/MKL) backend.
# All arguments are forwarded to the gprat_distributed binary, e.g.:
#   ./run_gprat_distributed.sh --start 128 --end 4096 --step 2 --tiles 8 --loop 3
#
# NOTE: this script only launches a single HPX locality. To launch multiple localities
# (e.g. one per process on the same node), run this binary directly N times with
# --hpx:localities=N --hpx:node=<index> instead of via this script, and see the
# "To run the distributed GPRat benchmark" section in the top-level README for a
# required workaround (HPX's TCP zero-copy serialization threshold) and the caveats
# for launching across multiple actual nodes.
#
# NOTE: the default Spack environment (gprat_cpu_gcc) builds HPX with networking=none,
# which rejects --hpx:localities outright. To build a binary that supports > 1 locality, this
# script defaults GPRAT_DIST_MULTI_LOCALITY=1 (set it to 0 beforehand to opt back into the
# single-locality build); on the simcl hosts this switches to the gprat_cpu_gcc_dist Spack
# environment (networking=tcp, OpenBLAS-only, see
# spack-repo/environments/setup_gprat_cpu_gcc_dist.sh). Since the shared scratch Spack instance
# on those hosts is owned by another account, GPRAT_DIST_MULTI_LOCALITY=1 skips sourcing it and
# instead uses whatever `spack` is already on the user's own PATH (e.g. a personal Spack
# install with its own gprat_cpu_gcc_dist environment).
#
# With GPRAT_DIST_MULTI_LOCALITY=1, the script itself launches one run per locality count in
# GPRAT_DIST_LOCALITIES (default "1 2 4"), spawning the N processes (--hpx:localities=N
# --hpx:node=0..N-1) each round instead of a single-locality invocation.

set -e # Exit immediately if a command exits with a non-zero status.

: "${GPRAT_DIST_MULTI_LOCALITY:=1}"

is_simcl_host() {
  case " simcl1n1 simcl1n2 simcl1n3 simcl1n4 " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

###################################################################################################
# Set Spack if on simcl1n1, simcl1n2, simcl1n3, or simcl1n4
###################################################################################################

if [[ "$GPRAT_DIST_MULTI_LOCALITY" != "1" ]] && is_simcl_host "$HOSTNAME"; then

  spack_destination="/scratch-simcl1/grafml/Programs/spack-fp2-simcl1n1"
  source $spack_destination/spack/share/spack/setup-env.sh

fi

###################################################################################################
# Setup environment depending on the host
###################################################################################################

if command -v spack &>/dev/null; then

  echo "Spack command found, checking for environments..."

  HOSTNAME=$(hostname -s)

  # ipvs-epyc1 ####################################################################################
  if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then

    if spack env list | grep -q "gprat_cpu_gcc"; then
      echo "Found gprat_cpu_gcc environment, activating it."
      spack env activate gprat_cpu_gcc
      module load gcc/14.2.0
      export CXX=g++
      export CC=gcc
    fi

  # sven0 and sven1 ###############################################################################
  elif [[ "$HOSTNAME" == "sven0" || "$HOSTNAME" == "sven1" ]]; then

    spack load openblas arch=linux-fedora38-riscv64
    HPX_CMAKE=$HOME/git_workspace/build-scripts/build/hpx/lib64/cmake/HPX
    export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/hpx/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/boost/lib:$LD_LIBRARY_PATH
    export LD_PRELOAD=$HOME/git_workspace/build-scripts/build/jemalloc/lib/libjemalloc.so.2

  # aarch64 #######################################################################################
  elif [[ $(uname -i) == "aarch64" ]]; then

    spack load gcc@14.2.0
    if spack env list | grep -q "gprat_cpu_arm"; then
      echo "Found gprat_cpu_arm environment, activating it."
      spack env activate gprat_cpu_arm
    fi

  # simcl1n1, simcl1n2, simcl1n3, simcl1n4 (CPU only) #############################################
  elif is_simcl_host "$HOSTNAME"; then

    if [[ "$GPRAT_DIST_MULTI_LOCALITY" == "1" ]]; then

      if spack env list | grep -q "gprat_cpu_gcc_dist"; then
        echo "Found gprat_cpu_gcc_dist environment, activating it."
        spack env activate gprat_cpu_gcc_dist
        module load gcc/14.1.0
        export CXX=g++
        export CC=gcc
        # No MKL variant is maintained for this environment; build against OpenBLAS.
        GPRAT_ENABLE_MKL_ARGS=(-DGPRAT_ENABLE_MKL=OFF)
        LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH
        LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH
      else
        echo "Cannot find Spack environment gprat_cpu_gcc_dist. Please run spack-repo/environments/setup_gprat_cpu_gcc_dist.sh" 1>&2
        exit 1
      fi

    elif spack env list | grep -q "gprat_cpu_gcc"; then
      echo "Found gprat_cpu_gcc environment, activating it."
      spack env activate gprat_cpu_gcc
      module load gcc/14.1.0
      export CXX=g++
      export CC=gcc
      LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH
      LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH
      LD_LIBRARY_PATH=$(spack location -i intel-oneapi-mkl)/lib:$LD_LIBRARY_PATH
    else
      echo "Cannot find Spack environment gprat_cpu_gcc. Please run spack-repo/environments/setup_gprat_cpu_gcc.sh" 1>&2
      exit 1
    fi

  # unknown host ##################################################################################
  else

    echo "Caution: This script does not cover host $HOSTNAME."

  fi

else

  echo "Spack command not found. Building example without Spack."

fi

###################################################################################################
# Configure APEX
###################################################################################################

export APEX_SCREEN_OUTPUT=0
export APEX_DISABLE=1

###################################################################################################
# Compile code
###################################################################################################

# Unlike examples/gprat_cpp, examples/gprat_distributed is only ever built in-tree
# (it has no standalone/out-of-tree CMake support), so we build it as part of the
# main GPRat build with GPRAT_WITH_DISTRIBUTED enabled.

# Resolve the script's own directory so cmake paths are always correct
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPRAT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$GPRAT_ROOT"

HPX_DIR_ARGS=()
if [[ -n "$HPX_CMAKE" ]]; then
  HPX_DIR_ARGS=(-DHPX_DIR="$HPX_CMAKE")
fi

# Multi-locality builds use a distinct Spack toolchain (networking=tcp HPX, OpenBLAS-only)
# from the default single-locality build. Building both into the same build/release-linux
# directory poisons the CMake cache with paths from whichever toolchain configured it last
# (e.g. linking against one env's HPX headers while another env's .so is on
# LD_LIBRARY_PATH), so give multi-locality builds their own build directory.
BUILD_DIR="build/release-linux"
if [[ "$GPRAT_DIST_MULTI_LOCALITY" == "1" ]]; then
  BUILD_DIR="build/release-linux-dist"
fi

cmake --preset release-linux -B "$BUILD_DIR" -DGPRAT_WITH_DISTRIBUTED=ON "${HPX_DIR_ARGS[@]}" "${GPRAT_ENABLE_MKL_ARGS[@]}"
cmake --build "$BUILD_DIR" --target gprat_distributed -j

###################################################################################################
# Run code
###################################################################################################

GPRAT_DISTRIBUTED_BIN="$GPRAT_ROOT/$BUILD_DIR/examples/gprat_distributed/gprat_distributed"

if [[ "$GPRAT_DIST_MULTI_LOCALITY" == "1" ]]; then

  # Run from GPRAT_ROOT so the default data/data_1024/... paths resolve.
  for N in ${GPRAT_DIST_LOCALITIES:-1 2 4}; do

    echo "Running GPRat distributed benchmark ($N locality/localities)"

    pids=()
    "$GPRAT_DISTRIBUTED_BIN" --hpx:localities="$N" --hpx:node=0 \
      --hpx:ini=hpx.parcel.zero_copy_serialization_threshold=999999999 "$@" &
    pids+=($!)
    for ((node = 1; node < N; node++)); do
      "$GPRAT_DISTRIBUTED_BIN" --hpx:localities="$N" --hpx:node="$node" \
        --hpx:ini=hpx.parcel.zero_copy_serialization_threshold=999999999 &
      pids+=($!)
    done
    # `wait pid1 pid2 ...` only reports the exit status of the LAST pid, which would
    # let a crash of an earlier locality (e.g. node 0, the console process) slip past
    # `set -e`. Wait on each PID individually and fail if any of them failed.
    failed=0
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
    if [[ "$failed" != "0" ]]; then
      echo "At least one locality process failed ($N locality/localities run)" 1>&2
      exit 1
    fi

    echo "Finished running GPRat distributed benchmark ($N locality/localities)"

  done

else

  echo "Running GPRat distributed benchmark (single locality)"

  # Run from GPRAT_ROOT so the default data/data_1024/... paths resolve.
  "$GPRAT_DISTRIBUTED_BIN" "$@"

  echo "Finished running GPRat distributed benchmark"

fi
