# Inner CMakeLists.txt

# Options
# ##############################################################################

# CUDA and SYCL are mutually exclusive, catch this error case
if(GPRAT_WITH_CUDA AND GPRAT_WITH_SYCL)
  message(
    FATAL_ERROR
      "GPRAT_WITH_CUDA and GPRAT_WITH_SYCL cannot be enabled at the same time.")
endif()

# SYCL
option(GPRAT_WITH_SYCL "Enable SYCL support with oneMath" OFF)

option(GPRAT_SYCL_NVIDIA "Enable NVIDIA GPU support with oneMath SYCL backend"
       OFF)
option(GPRAT_SYCL_AMD "Enable AMD GPU support with oneMath SYCL backend" OFF)
option(GPRAT_SYCL_INTEL "Enable Intel GPU support with oneMath SYCL backend"
       OFF)

set(HIP_TARGETS
    ""
    CACHE STRING "Target HIP architectures")

# CUDA
option(GPRAT_WITH_CUDA "Enable GPU support with CUDA, cuSolver, cuBLAS" OFF)

add_compile_definitions(GPRAT_WITH_CUDA=$<BOOL:${GPRAT_WITH_CUDA}>
                        GPRAT_WITH_SYCL=$<BOOL:${GPRAT_WITH_SYCL}>)

if(GPRAT_WITH_CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  enable_language(CUDA)
endif()

# Source files
# ##############################################################################

# Option for steps duration measurement with APEX
option(GPRAT_APEX_STEPS "Enable measuring duration of steps with APEX" OFF)
# Pass variable to C++ code
add_compile_definitions(GPRAT_APEX_STEPS=$<BOOL:${GPRAT_APEX_STEPS}>)

# Option for measuring duration of assembly of covariance matrix and right
# looking cholesky in the cholesky function using APEX. Technically, the
# cholesky function also returns a large matrix but this adds significant
# overhead to the whole function call and does not provide much useful
# information on the optimized assembly and right looking cholesky steps which
# are used by other GP operations.
option(
  GPRAT_APEX_CHOLESKY
  "Enable measuring duration of cholesky assembly and computation with APEX"
  OFF)
# Pass variable to C++ code
add_compile_definitions(GPRAT_APEX_CHOLESKY=$<BOOL:${GPRAT_APEX_CHOLESKY}>)

# Set general GPRAT source files
set(SOURCE_FILES
    src/gprat.cpp
    src/utils.cpp
    src/performance_counters.cpp
    src/target.cpp
    src/tile_data.cpp
    src/kernels.cpp
    src/hyperparameters.cpp
    src/cpu/gp_functions.cpp
    src/cpu/gp_algorithms.cpp
    src/cpu/gp_uncertainty.cpp
    src/cpu/gp_optimizer.cpp
    src/cpu/tiled_algorithms.cpp
    src/cpu/adapter_cblas_fp32.cpp
    src/cpu/adapter_cblas_fp64.cpp)

# Add SYCL-specific source files if enabled
if(GPRAT_WITH_SYCL)
  list(
    APPEND
    SOURCE_FILES
    src/gpu/sycl/adapter_onemath.cpp
    src/gpu/sycl/sycl_gp_algorithms.cpp
    src/gpu/sycl/sycl_gp_functions.cpp
    src/gpu/sycl/sycl_gp_optimizer.cpp
    src/gpu/sycl/sycl_gp_uncertainty.cpp
    src/gpu/sycl/sycl_tiled_algorithms.cpp)
endif()

# Distributed HPX sources (HPX actions, tile cache, tiled dataset)
option(GPRAT_WITH_DISTRIBUTED "Enable distributed GP support via HPX actions"
       OFF)
add_compile_definitions(
  GPRAT_WITH_DISTRIBUTED=$<BOOL:${GPRAT_WITH_DISTRIBUTED}>)

if(GPRAT_WITH_DISTRIBUTED)
  list(
    APPEND
    SOURCE_FILES
    src/cpu/adapter_cblas_fp64_actions.cpp
    src/cpu/gp_algorithms_actions.cpp
    src/cpu/gp_uncertainty_actions.cpp
    src/cpu/gp_optimizer_actions.cpp
    src/tile_cache.cpp
    src/tiled_dataset.cpp)

  # HPX registration macros create static factory objects that icpx sees as
  # unused
  set_source_files_properties(
    src/cpu/adapter_cblas_fp64_actions.cpp src/cpu/gp_algorithms_actions.cpp
    src/cpu/gp_uncertainty_actions.cpp src/cpu/gp_optimizer_actions.cpp
    src/tiled_dataset.cpp PROPERTIES COMPILE_OPTIONS "-Wno-unused-variable")
endif()

if(GPRAT_WITH_CUDA)
  list(
    APPEND
    SOURCE_FILES
    src/gpu/cuda/gp_functions.cu
    src/gpu/cuda/gp_algorithms.cu
    src/gpu/cuda/gp_uncertainty.cu
    src/gpu/cuda/gp_optimizer.cu
    src/gpu/cuda/tiled_algorithms.cu
    src/gpu/cuda/adapter_cublas.cu
    src/gpu/cuda/cuda_kernels.cu)
endif()

# Add GPRat core library
# ##############################################################################

# Add GPRAT core library
add_library(gprat_core STATIC ${SOURCE_FILES})

# We'd like to consume it just as core under the GPRat namespace
set_property(TARGET gprat_core PROPERTY EXPORT_NAME core)
add_library(GPRat::core ALIAS gprat_core)

# Headers
# ##############################################################################

# Add them as PRIVATE sources here so they show up in project files Can't use
# PUBLIC etc. see: https://stackoverflow.com/a/62465051

file(GLOB_RECURSE header_files CONFIGURE_DEPENDS include/*.hpp
     include/cpu/*.hpp)

if(GPRAT_WITH_SYCL)
  file(GLOB_RECURSE sycl_headers CONFIGURE_DEPENDS include/sycl/*.hpp)
  list(APPEND header_files ${sycl_headers})
endif()

if(GPRAT_WITH_CUDA)
  file(GLOB_RECURSE cuda_headers CONFIGURE_DEPENDS include/gpu/*.cuh)
  list(APPEND header_files ${cuda_headers})
endif()

target_sources(gprat_core PRIVATE ${header_files})

# Linking
# ##############################################################################

# Link HPX libraries
target_link_libraries(gprat_core PUBLIC HPX::hpx)

# Link CUDA libaries if enabled
if(GPRAT_WITH_CUDA)
  find_package(CUDAToolkit MODULE REQUIRED)
  target_link_libraries(gprat_core PUBLIC CUDA::cusolver CUDA::cublas)
  # Flag not working for CLANG CUDA
  target_compile_features(gprat_core PUBLIC cuda_std_${CMAKE_CUDA_STANDARD})
endif()

# SYCL / oneMath linking
if(GPRAT_WITH_SYCL)

  find_package(oneMath REQUIRED)

  # Select SYCL backend architecture
  set(GPRAT_SYCL_CUDA_PATH
      ""
      CACHE PATH "CUDA installation path for icpx NVIDIA SYCL target.")

  set(GPRAT_SYCL_NVIDIA_ARCH
      ""
      CACHE
        STRING
        "NVIDIA GPU SM arch for SYCL (e.g. sm_80). Derived from GPRAT_SYCL_CUDA_PATH if empty."
  )

  set(SYCL_FLAGS ${SYCL_FLAGS} -Wno-unused-command-line-argument)

  if(GPRAT_SYCL_NVIDIA)

    set(SYCL_FLAGS ${SYCL_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
    if(GPRAT_SYCL_CUDA_PATH)
      list(APPEND SYCL_FLAGS
           "SHELL:-Xsycl-target-backend --cuda-path=${GPRAT_SYCL_CUDA_PATH}")
    endif()
    if(GPRAT_SYCL_NVIDIA_ARCH)
      list(
        APPEND SYCL_FLAGS
        "SHELL:-Xsycl-target-backend --cuda-gpu-arch=${GPRAT_SYCL_NVIDIA_ARCH}")
    endif()

  elseif(GPRAT_SYCL_AMD)

    set(SYCL_FLAGS ${SYCL_FLAGS} -fsycl -fsycl-targets=amdgcn-amd-amdhsa
                   -Xsycl-target-backend --offload-arch=${HIP_TARGETS})

  elseif(GPRAT_SYCL_INTEL)

    set(SYCL_FLAGS ${SYCL_FLAGS} -fsycl -fsycl-targets=spir64)
    target_compile_definitions(gprat_core PUBLIC GPRAT_SYCL_INTEL_GPU)

  else()

    message(
      FATAL_ERROR
        "No SYCL backend specified. Please enable one of GPRAT_SYCL_NVIDIA, GPRAT_SYCL_AMD, or GPRAT_SYCL_INTEL."
    )

  endif()

  target_compile_options(gprat_core PUBLIC ${SYCL_FLAGS})

  target_link_options(gprat_core PUBLIC ${SYCL_FLAGS})

  # Set link libraries based on SYCL backend
  if(GPRAT_SYCL_NVIDIA)

    target_link_libraries(gprat_core PUBLIC ONEMATH::onemath_blas_cublas
                                            ONEMATH::onemath_lapack_cusolver)

  elseif(GPRAT_SYCL_AMD)

    target_link_libraries(gprat_core PUBLIC ONEMATH::onemath_blas_rocblas
                                            ONEMATH::onemath_lapack_rocsolver)

  elseif(GPRAT_SYCL_INTEL)

    target_link_libraries(gprat_core PUBLIC ONEMATH::onemath_blas_mklgpu
                                            ONEMATH::onemath_lapack_mklgpu)

  else()

    message(
      FATAL_ERROR
        "No SYCL backend specified. Please enable one of GPRAT_SYCL_NVIDIA, GPRAT_SYCL_AMD, or GPRAT_SYCL_INTEL."
    )

  endif()

  if(TARGET ONEMATH::onemath)
    target_link_libraries(gprat_core PUBLIC ONEMATH::onemath)
  endif()

endif()

# Include directories
target_include_directories(
  gprat_core PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>")

# Link BLAS
if(GPRAT_ENABLE_MKL)
  # Link Intel oneMKL
  target_link_libraries(gprat_core PUBLIC MKL::mkl_intel_lp64 MKL::mkl_core
                                          MKL::MKL MKL::mkl_sequential)
  target_compile_definitions(gprat_core PUBLIC GPRAT_ENABLE_MKL)
else()
  # Link OpenBLAS
  target_link_libraries(gprat_core PUBLIC ${OpenBLAS_LIB})
endif()

target_compile_definitions(gprat_core
                           PUBLIC GPRAT_WITH_CUDA=$<BOOL:${GPRAT_WITH_CUDA}>)
if(GPRAT_ENABLE_BENCHMARK_CACHE_EVICTIONS)
  target_compile_definitions(gprat_core
                             PUBLIC GPRAT_ENABLE_BENCHMARK_CACHE_EVICTIONS)
endif()
target_compile_features(gprat_core PUBLIC cxx_std_20)

set_property(TARGET gprat_core PROPERTY POSITION_INDEPENDENT_CODE ON)

# Installation
# ##############################################################################

if(NOT CMAKE_SKIP_INSTALL_RULES)
  # We need to manually install those into CMAKE_INSTALL_INCLUDEDIR. Below
  # install(TARGETS ...) only setups the paths for the exported targets.
  install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT Development)

  install(
    TARGETS gprat_core
    EXPORT GPRatTargets
    RUNTIME COMPONENT Runtime
    LIBRARY COMPONENT Runtime NAMELINK_COMPONENT Development
    ARCHIVE COMPONENT Development
    INCLUDES
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

# End of file
# ##############################################################################
