# GPRat: Gaussian Process Regression using Asynchronous Tasks

<img align="right" width="15%" src="data/images/ratward_icon.jpg">
GPRat is an open-source library for Gaussian Process Regression.
Leveraging the asynchronous many-task runtime HPX, we aim to combine the performance of asynchronous parallelism in C++
with the ease of use of commonly available Python libraries.
Thus, GPRat can be conveniently integrated into Python projects without binding overheads or used directly with pure C++
code.
Computations run on CPUs as well as NVIDIA GPUs (CUDA) and Intel/AMD GPUs (SYCL), in single (fp32) and double (fp64)
precision.
GPRat further provides a NUMA-aware allocator for tile data, performance counters, and optional distributed execution
via HPX actions.

## Dependencies

GPRat depends on [HPX](https://hpx-docs.stellar-group.org/latest/html/index.html) for asynchronous task-based parallelization.
Furthermore, for CPU-only BLAS computation GPRat requires [OpenBLAS](http://www.openmathlib.org/OpenBLAS/) or [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html).
A [CUDA](https://developer.nvidia.com/cuda-toolkit) installation is required for GPU-only BLAS computations on NVIDIA hardware.
For GPU computations on Intel and AMD hardware, GPRat supports [SYCL](https://www.khronos.org/sycl/) via [oneMath](https://github.com/uxlfoundation/oneMath).

### Install dependencies

All dependencies can be installed using [Spack](https://github.com/spack/spack).
A script to install and setup spack for `GPRat` is provided in [`spack-repo`](spack-repo).
Spack environment configurations and setup scripts for CPU and GPU use are provided in
[`spack-repo/environments`](spack-repo/environments).

Since Spack is not available on Windows, we also support dependency installation using vcpkg.
For now, vcpkg builds are only tested on Windows.

## How To Compile

GPRat makes use of [CMake presets][1] to simplify the process of configuring the project.

For example, building and testing this project on a Linux machine is as easy as running the following commands:

```sh
cmake --preset=dev-linux
cmake --build --preset=dev-linux
ctest --preset=dev-linux
```

As a developer, you may create a `CMakeUserPresets.json` file at the root of the project that contains additional
presets local to your machine.
In addition to the build configuration `dev-linux`, there are `release-linux`, `dev-linux-cuda`, `release-linux-cuda`, `dev-linux-sycl`, and `release-linux-sycl`.
The configurations suffixed with `-cuda` build the library with CUDA for NVIDIA GPUs, and those suffixed with `-sycl` build it with SYCL support for Intel and AMD GPUs.

GPRat can be build with or without Python bindings.
The following options can be set to include / exclude parts of the project:

| Option name                    | Description                                                                          | Default value   |
|--------------------------------|--------------------------------------------------------------------------------------|-----------------|
| GPRAT_BUILD_CORE               | Enable/Disable building of the core library                                          | ON              |
| GPRAT_BUILD_BINDINGS           | Enable/Disable building of the Python bindings                                       | ON              |
| GPRAT_ENABLE_EXAMPLES          | Enable/Disable example projects                                                      | ON if top-level |
| GPRAT_ENABLE_TESTS             | Enable/Disable building of unit and integration tests                                | ON if top-level |
| GPRAT_ENABLE_FORMAT_TARGETS    | Enable/Disable code formatting helper targets                                        | ON if top-level |
| GPRAT_ENABLE_BENCHMARK_CACHE_EVICTIONS | Enable/Disable evicting data from caches before running BLAS operations      | ON              |
| GPRAT_ENABLE_MKL               | Enable/Disable support for Intel oneMKL                                              | OFF             |
| GPRAT_WITH_CUDA                | Enable/disable compilation with CUDA support (NVIDIA GPUs)                           | OFF             |
| GPRAT_WITH_SYCL                | Enable/disable compilation with SYCL support (Intel and AMD GPUs via oneMath)        | OFF             |
| GPRAT_WITH_DISTRIBUTED         | Enable/disable distributed GP support via HPX actions                                | OFF             |
| GPRAT_TEST_MULTI_LOCALITY      | Enable/disable CTest smoke tests that launch the benchmark across multiple localities | OFF             |
| GPRAT_APEX_STEPS               | Enable/disable compilation for steps duration measurement with APEX                  | OFF             |
| GPRAT_APEX_CHOLESKY            | Enable/disable compilation for measuring cholesky assembly and computation with APEX | OFF             |

A convenience script `compile_gprat.sh` is provided to configure, build, and install GPRat with a single command.
It takes five parameters:

```sh
./compile_gprat.sh [python/cpp] [cpu/cuda/sycl] [release/dev] [mkl/none] [steps/cholesky/none]
```

- `$1`: build the Python bindings (`python`) or the C++ library (`cpp`)
- `$2`: backend, CPU (`cpu`), CUDA for NVIDIA GPUs (`cuda`), or SYCL for Intel and AMD GPUs (`sycl`)
- `$3`: build in `release` or `dev` mode
- `$4`: enable Intel oneMKL (`mkl`) or use OpenBLAS (`none`)
- `$5`: APEX profiling, measure step durations (`steps`), cholesky assembly and computation (`cholesky`), or disable profiling (`none`)

Computations are supported in both single (fp32) and double (fp64) precision.

We also provide a spack package for GPRat in [`spack-repo/packages`](spack-repo/packages) for portable and convenient compilation. When the repository is added to spack, GPRat can be installed with `spack install gprat~cuda~bindings~examples blas=mkl` (or `blas=openblas`)

## How To Run

GPRat contains several examples. One to run the C++ code, one to run the Python code as well as two reference
implementations based on TensorFlow ([GPflow](https://github.com/GPflow/GPflow)) and PyTorch
([GPyTorch](https://github.com/cornellius-gp/gpytorch)).

### To run the GPRat C++ code

- Go to [`examples/gprat_cpp`](examples/gprat_cpp/)
- Set parameters in [`config.json`](examples/gprat_cpp/config.json)
- The example is built as part of the main project.
  - Execute `./build/<preset>/examples/gprat_cpp/gprat_cpp [--use-gpu]` (e.g. `build/dev-linux/...` for the
    `dev-linux` preset) to run the example.
  - If you want to use an installed GPRat version:
    Run `./run_gprat_cpp.sh [cpu/cuda/sycl] [nvidia/amd/intel]` to build and run the example.
    The second parameter selects the SYCL device and is only required when GPRat was compiled with the SYCL backend.

### To run GPRat with Python

- Go to [`examples/gprat_python`](examples/gprat_python/)
- Set parameters in [`config.json`](examples/gprat_python/config.json)
- Run `./run_gprat_python.sh [cpu/cuda/sycl] [nvidia/amd/intel]` to run the example.
  The second parameter selects the SYCL device and is only required when GPRat was compiled with the SYCL backend.

### To run the distributed GPRat benchmark

- Go to [`examples/gprat_distributed`](examples/gprat_distributed/)
- Configure the main project with `-DGPRAT_WITH_DISTRIBUTED=ON` to build [`examples/gprat_distributed`](examples/gprat_distributed/).
- The example is a CLI-driven scaling benchmark (no `config.json`) rather than a single "run one example" tool,
  since it sweeps over training-set sizes rather than running one fixed configuration.
- Execute `./build/<preset>/examples/gprat_distributed/gprat_distributed [options]` (e.g.
  `build/release-linux-dist/...` for a multi-locality build, see below), or run
  `./run_gprat_distributed.sh [options]` to build and run it. Useful options:
  - `--start`/`--end`/`--step`: training-set sizes to sweep over (e.g. `--start 128 --end 4096 --step 2`)
  - `--tiles`, `--regressors`, `--n_test`, `--opt_iter`, `--loop`: problem size and repetition count
  - `--enabled`: bitmask to select which of cholesky/optimize/predict/predict_with_uncertainty/predict_with_full_cov to run
  - `--train_x_path`/`--train_y_path`/`--test_path`: point at a larger dataset (e.g. one generated via
    [`data/generators`](data/generators/)) for a real scaling study; the defaults point at the small `data/data_1024`
    correctness fixture
  - `--output_csv`: where per-run timings are appended (defaults to `examples/gprat_distributed/output.csv`,
    matching the other examples)
- By default (`GPRAT_DIST_MULTI_LOCALITY=1`, on unless you set it to `0` before running
  `run_gprat_distributed.sh`) the script runs across multiple localities on one node: it builds a
  networking-enabled binary and, for each locality count in `GPRAT_DIST_LOCALITIES` (default `"1 2 4"`),
  launches that many processes itself with `--hpx:localities=N --hpx:node=0..N-1` (node `0` is the console
  process that receives your CLI options; the others just join), waiting for each round to finish before
  moving to the next. All arguments you pass are forwarded to node `0`. Set `GPRAT_DIST_MULTI_LOCALITY=0` to
  opt back into a single-locality run against the default `gprat_cpu_gcc` build.
  **Important:** HPX's TCP parcelport zero-copy path (`hpx.parcel.zero_copy_serialization_threshold`,
  8192 bytes by default) reliably hangs once tile sizes exceed it in a multi-locality run, so the script
  always raises it (`--hpx:ini=hpx.parcel.zero_copy_serialization_threshold=999999999`) for these runs.
  Running across multiple actual nodes additionally requires cluster-specific network configuration
  (AGAS bootstrap addresses, hostfile/job-scheduler integration) not set up here.
  - The default Spack environment (`gprat_cpu_gcc`) builds HPX with `networking=none`, which rejects
    `--hpx:localities` outright. `GPRAT_DIST_MULTI_LOCALITY=1` instead uses the `gprat_cpu_gcc_dist`
    Spack environment (`networking=tcp`, OpenBLAS-only — see
    `spack-repo/environments/setup_gprat_cpu_gcc_dist.sh`) and builds into a separate
    `build/release-linux-dist` directory to avoid mixing the two toolchains.
  - Enable `-DGPRAT_TEST_MULTI_LOCALITY=ON` (in addition to `-DGPRAT_WITH_DISTRIBUTED=ON`) to register
    CTest smoke tests (`GPRat_test_distributed_multi_locality_{1,2,4}`) that launch `gprat_distributed`
    across 1/2/4 localities; off by default since it needs the same networking-enabled HPX build.

### To run GPflow reference

- Go to [`examples/gpflow_reference`](examples/gpflow_reference/)
- Set parameters in [`config.json`](examples/gpflow_reference/config.json)
- Run `./run_gpflow.sh [cpu/gpu/arm] [nvidia/amd/intel]` to run example.
  The second parameter selects the GPU vendor and is only required when `gpu` is specified.

### To run GPyTorch reference

- Go to [`examples/gpytorch_reference`](examples/gpytorch_reference/)
- Set parameters in [`config.json`](examples/gpytorch_reference/config.json)
- Run `./run_gpytorch.sh [cpu/gpu/arm] [nvidia/amd/intel]` to run example.
  The second parameter selects the GPU vendor and is only required when `gpu` is specified.

## The Team

The GPRat library is developed by the [Scientific Computing](https://www.ipvs.uni-stuttgart.de/departments/sc/)
department at IPVS at the University of Stuttgart.
The project is a joined effort of multiple undergraduate, graduate, and PhD students under the supervision of
[Prof. Dr. Dirk Pflüger](https://www.f05.uni-stuttgart.de/en/faculty/contactpersons/Pflueger-00005/).
We specifically thank the follow contributors:

- [Alexander Strack](https://www.ipvs.uni-stuttgart.de/de/institut/team/Strack-00001/):
  Maintainer and [initial framework](https://doi.org/10.1007/978-3-031-32316-4_5).

- [Maksim Helmann](https://de.linkedin.com/in/maksim-helmann-60b8701b1):
  [Optimization, Python bindings and reference implementations](https://doi.org/10.48550/arXiv.2505.00136).

- [Henrik Möllmann](https://www.linkedin.com/in/moellh/):
  [CUDA backend via cuBLAS/cuSOLVER](tbd.).

- [Marcel Graf](https://github.com/MarcelGraf0710):
  [SYCL backend via oneMath](tbd.).

- [Tim Niederhausen](https://github.com/timniederhausen):
  [Distributed GP via HPX actions](tbd.).

## How To Cite

```
@InProceedings{GPRat2025,
  author={Helmann, Maksim and Strack, Alexander and Pfl{\"u}ger, Dirk},
  title={{GPRat: Gaussian Process Regression with Asynchronous Tasks}},
  booktitle={Asynchronous Many-Task Systems and Applications},
  year={2025},
  publisher={Springer Nature}
}
```

[1]: https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html
