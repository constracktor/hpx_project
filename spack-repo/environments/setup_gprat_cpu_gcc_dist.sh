#!/usr/bin/env bash
set -e
# Script to setup a CPU spack environment for GPRat's multi-locality distributed
# benchmark using a recent gcc. Unlike setup_gprat_cpu_gcc.sh, this builds HPX with
# networking=tcp (required for --hpx:localities > 1) instead of networking=none, and
# uses OpenBLAS instead of MKL since no MKL variant of this environment is maintained.

# Load GCC compiler
module load gcc/14.1.0
env_name=gprat_cpu_gcc_dist

# Find GCC compiler with spack
spack_destination="/scratch-simcl1/grafml/Programs/spack-fp2-simcl1n1"
source $spack_destination/spack/share/spack/setup-env.sh
spack compiler find

# Create environment and copy config file
spack env create $env_name
cp spack_cpu_gcc_dist.yaml $spack_destination/spack/var/spack/environments/$env_name/spack.yaml
spack env activate $env_name

# Use external python
spack external find python

# setup environment
spack concretize -f
spack install
