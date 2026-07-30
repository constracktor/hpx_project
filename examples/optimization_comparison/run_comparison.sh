#!/bin/bash
# Runs Adam hyperparameter optimization with GPRat, GPflow, and GPyTorch from
# the same starting point with matched Adam settings, and compares the
# fitted hyperparameters and loss trajectory. CPU only.
#
# Assumes examples/gpflow_reference/gpflow_cpu_env and
# examples/gpytorch_reference/gpytorch_cpu_env already exist (see
# run_gpflow.sh cpu / run_gpytorch.sh cpu) and that GPRat has been built via
# the release-linux CMake preset.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../site_paths.sh"
cd "$SCRIPT_DIR"

HOSTNAME=$(hostname -s)

### ENVIRONMENT SETUP #############################################################################

if [[ "$HOSTNAME" == "sven0" || "$HOSTNAME" == "sven1" ]]; then

	export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/hpx/lib64:$LD_LIBRARY_PATH
	export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/boost/lib:$LD_LIBRARY_PATH
	export LD_PRELOAD=$HOME/git_workspace/build-scripts/build/jemalloc/lib/libjemalloc.so.2

elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" || \
        "$HOSTNAME" == "simcl1n3" || "$HOSTNAME" == "simcl1n4" ]]; then

	source "$SIMCL1_SPACK_ROOT/spack/share/spack/setup-env.sh"

	if spack env list | grep -q "gprat_cpu_gcc"; then
		echo "Found gprat_cpu_gcc environment, activating it."
		spack env activate gprat_cpu_gcc
		module load gcc/14.1.0
		export LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH
		export LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH
		export LD_LIBRARY_PATH=$(spack location -i intel-oneapi-mkl)/lib:$LD_LIBRARY_PATH
	fi

elif [[ "$HOSTNAME" == "pcsgs04" ]]; then

	source "$PCSGS04_SPACK_ROOT/share/spack/setup-env.sh"

fi

### INSTALL MATCHING GPRAT BUILD ##################################################################

GPRAT_ROOT="$SCRIPT_DIR/../.."
cmake --install "$GPRAT_ROOT/build/release-linux" --prefix "$SCRIPT_DIR"
cp "$GPRAT_ROOT/build/release-linux/bindings"/gprat.cpython-*.so "$SCRIPT_DIR/lib/"

### EXECUTION #####################################################################################

python3 compare.py
