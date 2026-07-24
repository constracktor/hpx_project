#!/bin/bash
# Input $1: Specify how GPRat was compiled, options:	cpu/cuda/sycl
# Input $2: If GPRat was compiled with SYCL backend:	nvidia/amd/intel

set -e # Exit immediately if a command exits with a non-zero status.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../site_paths.sh"

# Get current hostname
HOSTNAME=$(hostname -s)

###################################################################################################
# Set GPU flag
###################################################################################################

if [[ -z "$1" ]]; then
  echo "Input parameter is missing. Using default: Run computations on CPU"
  use_gpu=""
elif [[ "$1" == "cuda" ]]; then
  use_gpu="--use-gpu"
  if [[ \
    "$HOSTNAME" != "simcl1n1" && \
    "$HOSTNAME" != "simcl1n2" && \
    "$HOSTNAME" != "simcl1n3" && \
    "$HOSTNAME" != "simcl1n4" ]];
  then
    echo "GPU execution with this script is only supported on simcl1n1, simcl1n2, simcl1n3, and simcl1n4." 1>&2
    exit 1
  fi
elif [[ "$1" == "sycl" ]]; then
  use_gpu="--use-gpu"
  if [[ \
    "$HOSTNAME" != "simcl1n1" && \
    "$HOSTNAME" != "simcl1n2" && \
    "$HOSTNAME" != "simcl1n3" && \
    "$HOSTNAME" != "simcl1n4" ]];
  then
    echo "GPU execution with this script is only supported on simcl1n1, simcl1n2, simcl1n3, and simcl1n4." 1>&2
    exit 1
  fi
elif [[ "$1" != "cpu" ]]; then
  echo "Please specify input parameter: cpu/cuda/sycl"
  exit 1
fi

###################################################################################################
# Set Spack if on simcl1n1, simcl1n2, simcl1n3, or simcl1n4
###################################################################################################

if [[ \
  "$HOSTNAME" == "simcl1n1" || \
  "$HOSTNAME" == "simcl1n2" || \
  "$HOSTNAME" == "simcl1n3" || \
  "$HOSTNAME" == "simcl1n4" ]];
then

  spack_destination="$SIMCL1_SPACK_ROOT"
  source $spack_destination/spack/share/spack/setup-env.sh

fi

# Set Spack if on pcsgs04
if [[ "$HOSTNAME" == "pcsgs04" ]]; then

  spack_destination="$PCSGS04_SPACK_ROOT"
  source $spack_destination/share/spack/setup-env.sh

fi

###################################################################################################
# Setup environment depending on the host
###################################################################################################

if command -v spack &>/dev/null; then

  echo "Spack command found, checking for environments..."

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
    GPRAT_LIB_SUFFIX=64

  # aarch64 #######################################################################################
  elif [[ $(uname -i) == "aarch64" ]]; then

    spack load gcc@14.2.0
    if spack env list | grep -q "gprat_cpu_arm"; then
      echo "Found gprat_cpu_arm environment, activating it."
      spack env activate gprat_cpu_arm
    fi
    GPRAT_LIB_SUFFIX=64

  # simcl1n1 and simcl1n2 with NVIDIA GPUs ########################################################
  elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then

    if [[ "$1" == "cpu" ]]; then

      if spack env list | grep -q "gprat_cpu_gcc"; then
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

    else

      if spack env list | grep -q "gprat_gpu_clang"; then
        echo "Found gprat_gpu_clang environment, activating it."
        spack env activate gprat_gpu_clang
        module load cuda/12.0.1
        module load clang/17.0.1
        LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH
        LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH
        LD_LIBRARY_PATH=$(spack location -i intel-oneapi-mkl)/lib:$LD_LIBRARY_PATH
      else
        echo "Cannot find Spack environment gprat_gpu_clang. Please run spack-repo/environments/setup_gprat_gpu_clang.sh" 1>&2
        exit 1
      fi

      if [[ "$1" == "cuda" ]]; then

        export CXX=clang++
        export CC=clang

      elif [[ "$1" == "sycl" ]]; then

        # Source Intel oneAPI environment if icpx is not yet in PATH
        ONEAPI_COMPILER_ROOT=""
        if ! command -v icpx &>/dev/null; then
          ONEAPI_SETVARS="/import/sgs.scratch-simcl1/breyerml/Programs/spack/opt/spack/linux-zen4/intel-oneapi-compilers-2025.1.1-5ynklzzqslh265azbglzqdtecdghl7ob/setvars.sh"
          if [[ -f "$ONEAPI_SETVARS" ]]; then
            ONEAPI_COMPILER_ROOT="$(dirname $ONEAPI_SETVARS)/compiler/2025.1"
            export PATH="$ONEAPI_COMPILER_ROOT/bin:$PATH"
            export LD_LIBRARY_PATH="$ONEAPI_COMPILER_ROOT/lib:${LD_LIBRARY_PATH:-}"
          fi
        fi

        if command -v icpx &>/dev/null; then
          export CXX=icpx
          export CC=icx
          CMAKE_PREFIX_PATH="${ONEMATH_NVIDIA_ROOT}/lib/cmake/oneMath:${CMAKE_PREFIX_PATH:-}"
        else
          echo "Intel oneAPI DPC++ compiler (icpx) not found. Please make sure that icpx is available in your PATH." 1>&2
          exit 1
        fi

      fi

    fi

  # simcl1n3 with AMD GPU #########################################################################
  elif [[ "$HOSTNAME" == "simcl1n3" ]]; then

    if [[ "$1" == "cpu" ]]; then

      if spack env list | grep -q "gprat_cpu_gcc"; then
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

    else

      if spack env list | grep -q "gprat_gpu_clang"; then
        echo "Found gprat_gpu_clang environment, activating it."
        spack env activate gprat_gpu_clang
        LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH
        LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH
        LD_LIBRARY_PATH=$(spack location -i intel-oneapi-mkl)/lib:$LD_LIBRARY_PATH
      else
        echo "Cannot find Spack environment gprat_gpu_clang. Please run spack-repo/environments/setup_gprat_gpu_clang.sh" 1>&2
        exit 1
      fi

      if [[ "$1" == "sycl" ]]; then

        # Source Intel oneAPI environment if icpx is not yet in PATH
        ONEAPI_COMPILER_ROOT=""
        if ! command -v icpx &>/dev/null; then
          ONEAPI_SETVARS="/import/sgs.scratch-simcl1/breyerml/Programs/spack/opt/spack/linux-zen4/intel-oneapi-compilers-2025.1.1-5ynklzzqslh265azbglzqdtecdghl7ob/setvars.sh"
          if [[ -f "$ONEAPI_SETVARS" ]]; then
            ONEAPI_COMPILER_ROOT="$(dirname $ONEAPI_SETVARS)/compiler/2025.1"
            export PATH="$ONEAPI_COMPILER_ROOT/bin:$PATH"
            export LD_LIBRARY_PATH="$ONEAPI_COMPILER_ROOT/lib:${LD_LIBRARY_PATH:-}"
          fi
        fi
        if [[ -z "$ONEAPI_COMPILER_ROOT" ]] && command -v icpx &>/dev/null; then
          ONEAPI_COMPILER_ROOT="$(dirname $(dirname $(which icpx)))"
          export LD_LIBRARY_PATH="$ONEAPI_COMPILER_ROOT/lib:${LD_LIBRARY_PATH:-}"
        fi

        # Set up ROCm/HIP environment (required for AMD GPU device libraries at link and run time)
        ROCM_PATH=${ROCM_PATH:-/opt/rocm-6.4.0}
        if [[ -d "$ROCM_PATH" ]]; then
          export PATH="$ROCM_PATH/bin:$PATH"
          export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib:${LD_LIBRARY_PATH:-}"
          export LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib:${LIBRARY_PATH:-}"
          export ROCM_PATH
        fi
        # Compatibility shim: libamd_comgr.so.2 → libamd_comgr.so.3 for icpx HIP adapter
        COMGR_COMPAT_DIR="/data/scratch-simcl1/breyerml/Programs/.modulefiles/icpx"
        if [[ -d "$COMGR_COMPAT_DIR" ]]; then
          export LD_LIBRARY_PATH="$COMGR_COMPAT_DIR:${LD_LIBRARY_PATH:-}"
        fi
        export HSA_XNACK=1

        if command -v icpx &>/dev/null; then
          export CXX=icpx
          export CC=icx
          CMAKE_PREFIX_PATH="${ONEMATH_AMD_ROOT}/lib/cmake/oneMath:${CMAKE_PREFIX_PATH:-}"
        else
          echo "Intel oneAPI DPC++ compiler (icpx) not found. Please make sure that icpx is available in your PATH." 1>&2
          exit 1
        fi

      fi

    fi

  # simcl1n4 without GPU ##########################################################################
  elif [[ "$HOSTNAME" == "simcl1n4" ]]; then

    if [[ "$1" == "cuda" || "$1" == "sycl" ]]; then
      echo "Error: simcl1n4 does not have a GPU." 1>&2
      exit 1
    fi

    if spack env list | grep -q "gprat_cpu_gcc"; then
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

  # pcsgs04 with Intel GPU (Arc B580) #############################################################
  elif [[ "$HOSTNAME" == "pcsgs04" ]]; then

    if spack env list | grep -q "gprat_gpu_clang"; then

      echo "Found gprat_gpu_clang environment, activating it."
      spack env activate gprat_gpu_clang

      if [[ "$1" == "sycl" ]]; then

        # icpx isn't in the gprat_gpu_clang env; source it from the system
        # oneAPI install, pinned to 2025.3 (newer releases break this oneMath build).
        if ! command -v icpx &>/dev/null && [[ -f /opt/intel/oneapi/compiler/2025.3/env/vars.sh ]]; then
          source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
        fi

        # libumf is needed on LD_LIBRARY_PATH or GPU enumeration silently finds nothing.
        if [[ -f /opt/intel/oneapi/umf/latest/env/vars.sh ]]; then
          source /opt/intel/oneapi/umf/latest/env/vars.sh
        fi

        # oneMath's Level-Zero libs need system MKL 2025.3 (and matching TBB),
        # not the older MKL/TBB bundled in gprat_gpu_clang.
        if [[ -f /opt/intel/oneapi/mkl/2025.3/env/vars.sh ]]; then
          source /opt/intel/oneapi/mkl/2025.3/env/vars.sh
        fi
        if [[ -f /opt/intel/oneapi/tbb/latest/env/vars.sh ]]; then
          source /opt/intel/oneapi/tbb/latest/env/vars.sh
        fi

        # gprat_gpu_clang's bundled TBB is missing symbols oneMath's MKL needs;
        # prepend the matching oneAPI TBB to LIBRARY_PATH for the linker.
        if [[ -n "${TBBROOT:-}" ]]; then
          LIBRARY_PATH="$TBBROOT/lib:${LIBRARY_PATH:-}"
        fi

        if command -v icpx &>/dev/null; then
          export CXX=icpx
          export CC=icx
          CMAKE_PREFIX_PATH="${ONEMATH_INTEL_ROOT}:${CMAKE_PREFIX_PATH:-}"
        else
          echo "Intel oneAPI DPC++ compiler (icpx) not found. Please make sure that icpx is available in your PATH." 1>&2
          exit 1
        fi

      fi

    else

      echo \
        "Cannot find Spack environment gprat_gpu_clang. Please run spack-repo/environments/setup_gprat_gpu_clang.sh" 1>&2
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

# Resolve the script's own directory so cmake paths are always correct
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPRAT_ROOT="$(pwd)/../.."
LIB_DIR="$(pwd)/lib${GPRAT_LIB_SUFFIX}"

if [[ "$1" == "cuda" ]]; then
  GPRAT_WITH_CUDA=ON
  GPRAT_WITH_SYCL=OFF
  GPRAT_BUILD_DIR="$GPRAT_ROOT/build/release-linux-cuda"
elif [[ "$1" == "sycl" ]]; then
  GPRAT_WITH_CUDA=OFF
  GPRAT_WITH_SYCL=ON
  GPRAT_BUILD_DIR="$GPRAT_ROOT/build/release-linux-sycl"
else
  GPRAT_WITH_CUDA=OFF
  GPRAT_WITH_SYCL=OFF
  GPRAT_BUILD_DIR="$GPRAT_ROOT/build/release-linux"
fi

# Install the matching GPRat build so the lib dir always matches the backend.
# Use $(pwd) as prefix: cmake places cmake files at $PREFIX/lib/cmake/GPRat
# which matches GPRAT_DIR below.
cmake --install "$GPRAT_BUILD_DIR" --prefix "$(pwd)"

GPRAT_DIR="$LIB_DIR/cmake/GPRat"

if [[ ! -d build ]]; then
  mkdir -p build
  cd build

  SYCL_COMPILER_ARGS=()
  if [[ "$1" == "sycl" ]]; then
    SYCL_COMPILER_ARGS=(
      -DCMAKE_C_COMPILER="$(which icx)"
      -DCMAKE_CXX_COMPILER="$(which icpx)"
    )
  fi

  # On pcsgs04, gprat_gpu_clang's RPATH shadows oneMath's own TBB; force an
  # explicit rpath-link to the correct TBB (set above via TBBROOT) first.
  EXTRA_LINKER_FLAGS=""
  if [[ -n "${TBBROOT:-}" ]]; then
    EXTRA_LINKER_FLAGS="-Wl,-rpath-link,${TBBROOT}/lib"
  fi

  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPRat_DIR=$GPRAT_DIR \
    -DHPX_DIR=$HPX_CMAKE \
    -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
    -DGPRAT_WITH_CUDA=$GPRAT_WITH_CUDA \
    -DGPRAT_WITH_SYCL=$GPRAT_WITH_SYCL \
    -DGPRAT_APEX_STEPS=OFF \
    -DGPRAT_APEX_CHOLESKY=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="${EXTRA_LINKER_FLAGS}" \
    "${SYCL_COMPILER_ARGS[@]}"
else
  cd build
fi

make -j

###################################################################################################
# Run code
###################################################################################################

echo "Running GPRat C++ example"

end_cores=$(python3 -c "import json; print(json.load(open('../config.json'))['END_CORES'])")
core_count=$((end_cores * 2))

taskset -c 0-$core_count:2 ./gprat_cpp $use_gpu

echo "Finished running GPRat C++ example"
