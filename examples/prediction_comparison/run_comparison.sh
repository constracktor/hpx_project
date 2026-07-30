#!/bin/bash
# Input $1 (optional): cpu (default) / cuda / sycl -- which GPRat build to compare.
# Input $2: If $1 is sycl: nvidia/amd/intel.
#
# Runs GPRat, GPflow, and GPyTorch on the same data/hyperparameters (no
# optimization) and compares their predictions.
#
# GPflow/GPyTorch always run on CPU (via their own venvs); only GPRat's
# device varies. GPRat's CPU and GPU builds are never loaded in the same
# process tree -- each invocation of this script activates exactly one
# environment and installs exactly one build into lib/, matching how
# examples/gprat_python/run_gprat_python.sh runs GPRat (mixing two HPX/BLAS
# builds' libraries on LD_LIBRARY_PATH in the same process is untested and
# not worth risking).
#
# Assumes examples/gpflow_reference/gpflow_cpu_env and
# examples/gpytorch_reference/gpytorch_cpu_env already exist (see
# run_gpflow.sh cpu / run_gpytorch.sh cpu) and that GPRat has been built via
# the matching CMake preset (release-linux / release-linux-cuda /
# release-linux-sycl).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../site_paths.sh"
cd "$SCRIPT_DIR"

HOSTNAME=$(hostname -s)

if [[ -z "$1" ]]; then
	echo "Input parameter is missing. Using default: Run computations on CPU"
	GPU=""
elif [[ "$1" == "cuda" || "$1" == "sycl" ]]; then
	GPU="--gpu $1"
	if [[ \
		"$HOSTNAME" != "simcl1n1" && \
		"$HOSTNAME" != "simcl1n2" && \
		"$HOSTNAME" != "simcl1n3" && \
		"$HOSTNAME" != "simcl1n4" && \
		"$HOSTNAME" != "pcsgs04" ]];
	then
		echo "GPU execution with this script is only supported on simcl1n1, simcl1n2, simcl1n3, simcl1n4, and pcsgs04." 1>&2
		exit 1
	fi
elif [[ "$1" != "cpu" ]]; then
	echo "Please specify input parameter: cpu/cuda/sycl"
	exit 1
fi

### ENVIRONMENT SETUP #############################################################################

if [[ "$HOSTNAME" == "sven0" || "$HOSTNAME" == "sven1" ]]; then

	export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/hpx/lib64:$LD_LIBRARY_PATH
	export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/boost/lib:$LD_LIBRARY_PATH
	export LD_PRELOAD=$HOME/git_workspace/build-scripts/build/jemalloc/lib/libjemalloc.so.2

elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" || \
        "$HOSTNAME" == "simcl1n3" || "$HOSTNAME" == "simcl1n4" ]]; then

	source "$SIMCL1_SPACK_ROOT/spack/share/spack/setup-env.sh"

	if [[ "$1" == "cuda" || "$1" == "sycl" ]]; then

		# simcl1n4 does not have a GPU
		if [[ "$HOSTNAME" == "simcl1n4" ]]; then
			echo "Machine $HOSTNAME does not have a GPU but you selected GPU execution." 1>&2
			exit 1
		fi

		if spack env list | grep -q "gprat_gpu_clang"; then
			echo "Found gprat_gpu_clang environment, activating it."
			spack env activate gprat_gpu_clang
			export LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH
			export LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH
			export LD_LIBRARY_PATH=$(spack location -i intel-oneapi-mkl)/lib:$LD_LIBRARY_PATH
		fi

		if [[ "$1" == "cuda" || ( "$1" == "sycl" && "$2" == "nvidia" ) ]]; then
			module load cuda/12.0.1
			module load clang/17.0.1
		fi

		if [[ "$1" == "sycl" ]]; then

			if [[ "$2" == "nvidia" ]]; then

				ONEMATH_PATH="${ONEMATH_NVIDIA_ROOT}/lib/"
				export LD_LIBRARY_PATH="$ONEMATH_PATH:$LD_LIBRARY_PATH"

			elif [[ "$2" == "amd" ]]; then

				ONEMATH_PATH="${ONEMATH_AMD_ROOT}/lib/"
				export LD_LIBRARY_PATH="$ONEMATH_PATH:$LD_LIBRARY_PATH"

				ROCM_PATH=${ROCM_PATH:-/opt/rocm-6.4.0}
				if [[ -d "$ROCM_PATH" ]]; then
					export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib:$LD_LIBRARY_PATH"
					export ROCM_PATH
				fi

				COMGR_COMPAT_DIR="/data/scratch-simcl1/breyerml/Programs/.modulefiles/icpx"
				if [[ -d "$COMGR_COMPAT_DIR" ]]; then
					export LD_LIBRARY_PATH="$COMGR_COMPAT_DIR:$LD_LIBRARY_PATH"
				fi

				ONEAPI_SETVARS="/import/sgs.scratch-simcl1/breyerml/Programs/spack/opt/spack/linux-zen4/intel-oneapi-compilers-2025.1.1-5ynklzzqslh265azbglzqdtecdghl7ob/setvars.sh"
				if ! command -v icpx &>/dev/null && [[ -f "$ONEAPI_SETVARS" ]]; then
					ONEAPI_COMPILER_ROOT="$(dirname $ONEAPI_SETVARS)/compiler/2025.1"
					export PATH="$ONEAPI_COMPILER_ROOT/bin:$PATH"
					export LD_LIBRARY_PATH="$ONEAPI_COMPILER_ROOT/lib:$LD_LIBRARY_PATH"
				elif command -v icpx &>/dev/null; then
					ONEAPI_COMPILER_ROOT="$(dirname $(dirname $(which icpx)))"
					export LD_LIBRARY_PATH="$ONEAPI_COMPILER_ROOT/lib:$LD_LIBRARY_PATH"
				fi

				export HSA_XNACK=1

			elif [[ "$2" == "intel" ]]; then

				echo "Machine $HOSTNAME does not have an Intel GPU." 1>&2
				exit 1

			elif [[ "$2" != "nvidia" ]]; then

				echo "Please specify gpu vendor: nvidia/amd/intel"
				exit 1

			fi

		fi

	elif [[ "$1" == "cpu" || -z "$1" ]]; then

		if spack env list | grep -q "gprat_cpu_gcc"; then
			echo "Found gprat_cpu_gcc environment, activating it."
			spack env activate gprat_cpu_gcc
			module load gcc/14.1.0
			export LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH
			export LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH
			export LD_LIBRARY_PATH=$(spack location -i intel-oneapi-mkl)/lib:$LD_LIBRARY_PATH
		fi

	fi

elif [[ "$HOSTNAME" == "pcsgs04" ]]; then

	source "$PCSGS04_SPACK_ROOT/share/spack/setup-env.sh"

	if [[ "$1" == "cuda" || "$1" == "sycl" ]]; then

		if spack env list | grep -q "gprat_gpu_clang"; then

			echo "Found gprat_gpu_clang environment, activating it."
			spack env activate gprat_gpu_clang
			export LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH

			if [[ "$1" == "sycl" ]]; then

				if [[ "$2" != "intel" ]]; then
					echo "pcsgs04 only has an Intel GPU. Please specify gpu vendor: intel" 1>&2
					exit 1
				fi

				if ! command -v icpx &>/dev/null && [[ -f /opt/intel/oneapi/compiler/2025.3/env/vars.sh ]]; then
					source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
				fi

				if [[ -f /opt/intel/oneapi/umf/latest/env/vars.sh ]]; then
					source /opt/intel/oneapi/umf/latest/env/vars.sh
				fi

				if [[ -f /opt/intel/oneapi/mkl/2025.3/env/vars.sh ]]; then
					source /opt/intel/oneapi/mkl/2025.3/env/vars.sh
				fi
				export LD_LIBRARY_PATH="/opt/intel/oneapi/tbb/2022.3/lib/intel64/gcc4.8:$LD_LIBRARY_PATH"

				ONEMATH_PATH="${ONEMATH_INTEL_ROOT}/lib"
				export LD_LIBRARY_PATH="$ONEMATH_PATH:$LD_LIBRARY_PATH"

			fi

		else

			echo \
				"Cannot find Spack environment gprat_gpu_clang. Please run spack-repo/environments/setup_gprat_gpu_clang.sh" 1>&2
			exit 1

		fi

	fi

fi

### INSTALL MATCHING GPRAT BUILD ##################################################################

GPRAT_ROOT="$SCRIPT_DIR/../.."

if [[ "$1" == "cuda" ]]; then
	GPRAT_BUILD_DIR="$GPRAT_ROOT/build/release-linux-cuda"
elif [[ "$1" == "sycl" ]]; then
	GPRAT_BUILD_DIR="$GPRAT_ROOT/build/release-linux-sycl"
else
	GPRAT_BUILD_DIR="$GPRAT_ROOT/build/release-linux"
fi

cmake --install "$GPRAT_BUILD_DIR" --prefix "$SCRIPT_DIR"
cp "$GPRAT_BUILD_DIR"/bindings/gprat.cpython-*.so "$SCRIPT_DIR/lib/"

### EXECUTION #####################################################################################

python3 compare.py $GPU
