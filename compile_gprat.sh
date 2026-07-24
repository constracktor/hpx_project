#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/site_paths.sh"

# Get current hostname
HOSTNAME=$(hostname -s)

###################################################################################################
# Parameters
###################################################################################################
# $1: python/cpp
# $2: cpu/cuda/sycl
# $3: release/dev
# $4: mkl/none
# $5: apex profiling options: steps/cholesky/none
# $6: distributed support: dist/none
#     NOTE: "dist" requires an HPX build with networking enabled (e.g. networking=tcp).
#     The default Spack environment gprat_cpu_gcc builds HPX with networking=none; a
#     GPRAT_WITH_DISTRIBUTED=ON binary built against it hangs or rejects
#     --hpx:localities > 1 at runtime. Use the gprat_cpu_gcc_dist environment (see
#     spack-repo/environments/setup_gprat_cpu_gcc_dist.sh) for distributed builds.

###################################################################################################
# Bindings
###################################################################################################
if [[ "$1" == "python" ]]; then
  BINDINGS=ON
  INSTALL_DIR=$(pwd)/examples/gprat_python
elif [[ "$1" == "cpp" ]]; then
  BINDINGS=OFF
  INSTALL_DIR=$(pwd)/examples/gprat_cpp
else
  echo "Please specify input parameter: python/cpp"
  exit 1
fi

###################################################################################################
# CMake Presets
###################################################################################################
if [[ "$2" == "cpu" ]]; then
  if [[ "$3" == "release" ]]; then
    PRESET=release-linux
  elif [[ "$3" == "dev" ]]; then
    PRESET=dev-linux
  else
    echo "Input parameter for release or dev mode is missing. Using default: Build in Release mode"
    PRESET=release-linux
  fi
elif [[ "$2" == "cuda" ]]; then
  if [[ "$3" == "release" ]]; then
    PRESET=release-linux-cuda
  elif [[ "$3" == "dev" ]]; then
    PRESET=dev-linux-cuda
  else
    echo "Input parameter for release or dev mode is missing. Using default: Build in Release mode"
    PRESET=release-linux-cuda
  fi
elif [[ "$2" == "sycl" ]]; then
  if [[ "$3" == "release" ]]; then
    PRESET=release-linux-sycl
  elif [[ "$3" == "dev" ]]; then
    PRESET=dev-linux-sycl
  else
    echo "Input parameter for release or dev mode is missing. Using default: Build in Release mode"
    PRESET=release-linux-sycl
  fi
else
  echo "Input parameter is not any of {cpu,cuda,sycl}. Using default: CPU in release mode."
  PRESET=release-linux
fi

###################################################################################################
# Select BLAS library
###################################################################################################
if [[ "$4" == "mkl" ]]; then
  USE_MKL=ON
else
  USE_MKL=OFF
fi

# Select APEX profiling option
if [[ "$5" == "steps" ]]; then
  GPRAT_APEX_STEPS=ON
  GPRAT_APEX_CHOLESKY=OFF
elif [[ "$5" == "cholesky" ]]; then
  GPRAT_APEX_STEPS=OFF
  GPRAT_APEX_CHOLESKY=ON
else
  GPRAT_APEX_STEPS=OFF
  GPRAT_APEX_CHOLESKY=OFF
fi

###################################################################################################
# Select distributed support
###################################################################################################
if [[ "$6" == "dist" ]]; then
  GPRAT_WITH_DISTRIBUTED=ON
  echo "Distributed support enabled. Make sure the active HPX build has networking enabled" \
    "(e.g. the gprat_cpu_gcc_dist Spack environment); HPX built with networking=none cannot" \
    "run with --hpx:localities > 1."
else
  GPRAT_WITH_DISTRIBUTED=OFF
fi

###################################################################################################
# SYCL target defaults (overridden per host below)
###################################################################################################
GPRAT_SYCL_NVIDIA=${GPRAT_SYCL_NVIDIA:-OFF}
GPRAT_SYCL_AMD=${GPRAT_SYCL_AMD:-OFF}
GPRAT_SYCL_INTEL=${GPRAT_SYCL_INTEL:-OFF}
HIP_TARGETS=${HIP_TARGETS:-}

###################################################################################################
# Pick Spack installation depending on the host
###################################################################################################

# Set Spack if on simcl1n1, simcl1n2, simcl1n3, or simcl1n4
if [[ \
  "$HOSTNAME" == "simcl1n1" || \
  "$HOSTNAME" == "simcl1n2" || \
  "$HOSTNAME" == "simcl1n3" || \
  "$HOSTNAME" == "simcl1n4" ]]; 
then

  spack_destination="$SIMCL1_SPACK_ROOT"
  source $spack_destination/spack/share/spack/setup-env.sh

fi

# Set Spack if on psgs04
if [[ "$HOSTNAME" == "pcsgs04" ]]; then

  spack_destination="$PCSGS04_SPACK_ROOT"
  source $spack_destination/share/spack/setup-env.sh

fi

###################################################################################################
# Setup Compilation Requirements
###################################################################################################

# Assuming Spack is found #########################################################################
if command -v spack &>/dev/null; then
  
  echo "Spack command found, checking for environments..."

  # ipvs-epyc1 ####################################################################################
  if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then

    # Check whether the gprat_cpu_gcc environment exists
    if spack env list | grep -q "gprat_cpu_gcc"; then
      echo "Found gprat_cpu_gcc environment, activating it."
      spack env activate gprat_cpu_gcc
      module load gcc/14.2.0
      export CXX=g++
      export CC=gcc
    fi

  # sven0 and sven1 ###############################################################################
  elif [[ "$HOSTNAME" == "sven0" || "$HOSTNAME" == "sven1" ]]; then

    # module load gcc/13.2.1
    spack load openblas arch=linux-fedora38-riscv64
    HPX_CMAKE=$HOME/git_workspace/build-scripts/build/hpx/lib64/cmake/HPX
    USE_MKL=OFF

  # aarch64 #######################################################################################
  elif [[ $(uname -i) == "aarch64" ]]; then

    spack load gcc@14.2.0
    # Check if the gprat_cpu_arm environment exists
    if spack env list | grep -q "gprat_cpu_arm"; then
      echo "Found gprat_cpu_arm environment, activating it."
      spack env activate gprat_cpu_arm
    fi
    USE_MKL=OFF

  # simcl1n1 and simcl1n2 with NVIDIA GPUs ########################################################
  elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then

    if [[ "$2" == "cpu" ]]; then # CPU build

      # Check if the gprat_cpu_gcc environment exists
      if spack env list | grep -q "gprat_cpu_gcc"; then

        echo "Found gprat_cpu_gcc environment, activating it."
        spack env activate gprat_cpu_gcc

        # Load GCC 14.1.0
        module load gcc/14.1.0

        # Set default compiler to GCC
        export CXX=g++
        export CC=gcc

      else

        echo "Cannot find Spack environment gprat_cpu_gcc. Please run spack-repo/environments/setup_gprat_cpu_gcc.sh" 1>&2
        exit -1

      fi

    else # GPU build

      if spack env list | grep -q "gprat_gpu_clang"; then

        echo "Found gprat_gpu_clang environment, activating it."
        spack env activate gprat_gpu_clang

        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk -F '.' '{print $1$2}')

        if [[ "$2" == "cuda" ]]; then # GPRat on NVIDIA GPUs with CUDA

          # Load CUDA and Clang modules
          module load cuda/12.0.1
          module load clang/17.0.1

          # Set default compiler to clang
          export CXX=clang++
          export CC=clang

        elif [[ "$2" == "sycl" ]]; then # GPRat on NVIDIA GPUs with SYCL

          # Source Intel oneAPI environment if icpx is not yet in PATH
          ONEAPI_COMPILER_ROOT=""
          if ! command -v icpx &>/dev/null; then
            ONEAPI_SETVARS="/import/sgs.scratch-simcl1/breyerml/Programs/spack/opt/spack/linux-zen4/intel-oneapi-compilers-2025.1.1-5ynklzzqslh265azbglzqdtecdghl7ob/setvars.sh"
            if [[ -f "$ONEAPI_SETVARS" ]]; then
              # setvars.sh requires a login shell; source just the compiler bin directory instead
              ONEAPI_COMPILER_ROOT="$(dirname $ONEAPI_SETVARS)/compiler/2025.1"
              export PATH="$ONEAPI_COMPILER_ROOT/bin:$PATH"
              export LD_LIBRARY_PATH="$ONEAPI_COMPILER_ROOT/lib:${LD_LIBRARY_PATH:-}"
            fi
          fi
          if [[ -z "$ONEAPI_COMPILER_ROOT" ]] && command -v icpx &>/dev/null; then
            # icpx was already in PATH; derive root from its location
            ONEAPI_COMPILER_ROOT="$(dirname $(dirname $(which icpx)))"
          fi

          if command -v icpx &>/dev/null; then

            # Set default compiler to icpx
            export CXX=icpx
            export CC=icx

            # Set GPRat build options for SYCL on NVIDIA GPUs
            GPRAT_SYCL_NVIDIA=ON

            # Load CUDA so icpx can find libdevice for NVIDIA SYCL targets
            module load cuda/12.0.1
            GPRAT_SYCL_CUDA_PATH=${CUDA_HOME}
            # Detect GPU SM arch (e.g. sm_80 for A30); default to sm_80 if detection fails
            GPRAT_SYCL_NVIDIA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | sed 's/^/sm_/')
            GPRAT_SYCL_NVIDIA_ARCH=${GPRAT_SYCL_NVIDIA_ARCH:-sm_80}

            # Add oneMath installation to CMAKE_PREFIX_PATH
            CMAKE_PREFIX_PATH="${ONEMATH_NVIDIA_ROOT}/lib/cmake/oneMath:${CMAKE_PREFIX_PATH:-}"

          else

            echo \
              "Intel oneAPI DPC++ compiler (icpx) not found. " \
              "Please make sure that icpx is available in your PATH." 1>&2
            exit -1

          fi

        fi

      else

        echo \
          "Cannot find Spack environment gprat_gpu_clang." \
          "Please run spack-repo/environments/setup_gprat_gpu_clang.sh" 1>&2
        exit -1

      fi

    fi

  # simcl1n3 with AMD GPU #########################################################################
  elif [[ "$HOSTNAME" == "simcl1n3" ]]; then

    if [[ "$2" == "cpu" ]]; then # CPU build

      # Check if the gprat_cpu_gcc environment exists
      if spack env list | grep -q "gprat_cpu_gcc"; then

        echo "Found gprat_cpu_gcc environment, activating it."
        spack env activate gprat_cpu_gcc

        # Load GCC 14.1.0
        module load gcc/14.1.0

        # Set default compiler to GCC
        export CXX=g++
        export CC=gcc

      else

        echo "Cannot find Spack environment gprat_cpu_gcc. Please run spack-repo/environments/setup_gprat_cpu_gcc.sh" 1>&2
        exit -1

      fi

    else # GPU build

      # Check whether the gprat_gpu_clang environment exists
      if spack env list | grep -q "gprat_gpu_clang"; then

        echo "Found gprat_gpu_clang environment, activating it."
        spack env activate gprat_gpu_clang

        if [[ "$2" == "sycl" ]]; then # GPRat on AMD GPUs with SYCL

          # Source Intel oneAPI environment if icpx is not yet in PATH
          ONEAPI_COMPILER_ROOT=""
          if ! command -v icpx &>/dev/null; then
            ONEAPI_SETVARS="/import/sgs.scratch-simcl1/breyerml/Programs/spack/opt/spack/linux-zen4/intel-oneapi-compilers-2025.1.1-5ynklzzqslh265azbglzqdtecdghl7ob/setvars.sh"
            if [[ -f "$ONEAPI_SETVARS" ]]; then
              # setvars.sh requires a login shell; source just the compiler bin directory instead
              ONEAPI_COMPILER_ROOT="$(dirname $ONEAPI_SETVARS)/compiler/2025.1"
              export PATH="$ONEAPI_COMPILER_ROOT/bin:$PATH"
              export LD_LIBRARY_PATH="$ONEAPI_COMPILER_ROOT/lib:${LD_LIBRARY_PATH:-}"
            fi
          fi
          if [[ -z "$ONEAPI_COMPILER_ROOT" ]] && command -v icpx &>/dev/null; then
            # icpx was already in PATH; derive root from its location
            ONEAPI_COMPILER_ROOT="$(dirname $(dirname $(which icpx)))"
          fi

          # Set up ROCm/HIP environment (required for AMD GPU device libraries at link time)
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

            # Set default compiler to icpx
            export CXX=icpx
            export CC=icx

            # Set GPRat build options for SYCL on AMD GPUs
            GPRAT_SYCL_AMD=ON

            # Set GPRat HIP target for AMD Instinct MI210 GPU (required by icpx)
            HIP_TARGETS="gfx90a"

            # Add oneMath installation to CMAKE_PREFIX_PATH
            CMAKE_PREFIX_PATH="${ONEMATH_AMD_ROOT}/lib/cmake/oneMath:${CMAKE_PREFIX_PATH:-}"

          else

            echo "Intel oneAPI DPC++ compiler (icpx) not found. Please make sure that icpx is available in your PATH." 1>&2
            exit -1

          fi

        fi

      else

        echo "Cannot find Spack environment gprat_gpu_clang. Please run spack-repo/environments/setup_gprat_gpu_clang.sh" 1>&2
        exit -1

      fi

    fi

  # simcl1n4 without GPU ##########################################################################
  elif [[ "$HOSTNAME" == "simcl1n4" ]]; then

    if [[ "$2" == "cuda" || "$2" == "sycl" ]]; then 

      echo "Error: simcl1n4 does not have a GPU." 1>&2
      exit -1

    fi

    # Check if the gprat_cpu_gcc environment exists
    if spack env list | grep -q "gprat_cpu_gcc"; then

      echo "Found gprat_cpu_gcc environment, activating it."
      spack env activate gprat_cpu_gcc

      # Load GCC 14.1.0
      module load gcc/14.1.0

      # Set default compiler to GCC
      export CXX=g++
      export CC=gcc

    else

      echo "Cannot find Spack environment gprat_cpu_gcc. Please run spack-repo/environments/setup_gprat_cpu_gcc.sh" 1>&2
      exit -1

    fi

  # pcsgs04 with Intel GPU (Arc B580) #############################################################
  elif [[ "$HOSTNAME" == "pcsgs04" ]]; then

    # Check whether the gprat_gpu_clang environment exists
    if spack env list | grep -q "gprat_gpu_clang"; then

      echo "Found gprat_gpu_clang environment, activating it."
      spack env activate gprat_gpu_clang

      if [[ "$2" == "sycl" ]]; then # GPRat on Intel GPUs with SYCL

        # icpx isn't in the gprat_gpu_clang env here; source it from the system
        # oneAPI install, pinned to 2025.3 (newer releases break this oneMath build).
        if ! command -v icpx &>/dev/null && [[ -f /opt/intel/oneapi/compiler/2025.3/env/vars.sh ]]; then
          source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
        fi

        # libumf is needed on LD_LIBRARY_PATH or GPU enumeration silently finds nothing.
        if [[ -f /opt/intel/oneapi/umf/latest/env/vars.sh ]]; then
          source /opt/intel/oneapi/umf/latest/env/vars.sh
        fi

        if command -v icpx &>/dev/null; then

          # Set default compiler to icpx
          export CXX=icpx
          export CC=icx

          # Set GPRat build options for SYCL on Intel GPUs
          GPRAT_SYCL_INTEL=ON

          CMAKE_PREFIX_PATH="${ONEMATH_INTEL_ROOT}:${CMAKE_PREFIX_PATH:-}"

        else

          echo \
            "Intel oneAPI DPC++ compiler (icpx) not found. Please make sure that icpx is available in your PATH." 1>&2
          exit -1

        fi
      
      fi

    else

      echo \
        "Cannot find Spack environment gprat_gpu_clang. Please run spack-repo/environments/setup_gprat_gpu_clang.sh" 1>&2
      exit -1

    fi

  # invalid hostnames #############################################################################
  else

    echo "Caution: This script does not cover host $HOSTNAME."

  fi

# Assuming Spack is not found
else

  echo "Spack command not found. Building example without Spack."

fi

###################################################################################################
# Set up CMake
###################################################################################################

# CPU build
if [[ $PRESET == "release-linux" || $PRESET == "dev-linux" ]]; then

  cmake --preset $PRESET -Wno-dev \
    -DGPRAT_BUILD_BINDINGS=$BINDINGS \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
    -DHPX_DIR=$HPX_CMAKE \
    -DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
    -DGPRAT_ENABLE_MKL=$USE_MKL \
    -DGPRAT_APEX_STEPS=${GPRAT_APEX_STEPS} \
    -DGPRAT_APEX_CHOLESKY=${GPRAT_APEX_CHOLESKY} \
    -DGPRAT_WITH_DISTRIBUTED=$GPRAT_WITH_DISTRIBUTED \
    -DGPRAT_ENABLE_TESTS=ON \
    -DGPRAT_ENABLE_EXAMPLES=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# CUDA build
elif [[ $PRESET == "release-linux-cuda" || $PRESET == "dev-linux-cuda" ]]; then

  cmake --preset $PRESET -Wno-dev \
    -DGPRAT_BUILD_BINDINGS=$BINDINGS \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
    -DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
    -DGPRAT_ENABLE_MKL=$USE_MKL \
    -DGPRAT_APEX_STEPS=${GPRAT_APEX_STEPS} \
    -DGPRAT_APEX_CHOLESKY=${GPRAT_APEX_CHOLESKY} \
    -DGPRAT_WITH_DISTRIBUTED=$GPRAT_WITH_DISTRIBUTED \
    -DCMAKE_C_COMPILER=$(which clang) \
    -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DCMAKE_CUDA_COMPILER=$(which clang++) \
    -DCMAKE_CUDA_FLAGS=--cuda-path=${CUDA_HOME} \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    -DCMAKE_EXE_LINKER_FLAGS="-L${CUDA_HOME}/targets/x86_64-linux/lib" \
    -DGPRAT_ENABLE_TESTS=ON \
    -DGPRAT_ENABLE_EXAMPLES=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# SYCL build
elif [[ $PRESET == "release-linux-sycl" || $PRESET == "dev-linux-sycl" ]]; then

  cmake --preset $PRESET -Wno-dev \
    -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
    -DGPRAT_BUILD_BINDINGS=$BINDINGS \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
    -DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
    -DGPRAT_ENABLE_MKL=$USE_MKL \
    -DGPRAT_APEX_STEPS=${GPRAT_APEX_STEPS} \
    -DGPRAT_APEX_CHOLESKY=${GPRAT_APEX_CHOLESKY} \
    -DGPRAT_WITH_DISTRIBUTED=$GPRAT_WITH_DISTRIBUTED \
    -DCMAKE_C_COMPILER=$(which icx) \
    -DCMAKE_CXX_COMPILER=$(which icpx) \
    -DGPRAT_WITH_SYCL=ON \
    -DGPRAT_SYCL_NVIDIA=$GPRAT_SYCL_NVIDIA \
    -DGPRAT_SYCL_AMD=$GPRAT_SYCL_AMD \
    -DGPRAT_SYCL_INTEL=$GPRAT_SYCL_INTEL \
    -DHIP_TARGETS=$HIP_TARGETS \
    -DGPRAT_SYCL_CUDA_PATH=${GPRAT_SYCL_CUDA_PATH:-} \
    -DGPRAT_SYCL_NVIDIA_ARCH=${GPRAT_SYCL_NVIDIA_ARCH:-} \
    -DCMAKE_BUILD_RPATH="${GPRAT_SYCL_CUDA_PATH:-}/lib64;${ONEAPI_COMPILER_ROOT}/lib" \
    -DGPRAT_ENABLE_TESTS=ON \
    -DGPRAT_ENABLE_EXAMPLES=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
fi

###################################################################################################
# Compile code
###################################################################################################
cmake --build --preset $PRESET -- -j
cmake --install build/$PRESET

###################################################################################################
# Run tests
###################################################################################################
cd build/$PRESET
ctest --output-on-failure --no-tests=ignore -C Release -j 2
