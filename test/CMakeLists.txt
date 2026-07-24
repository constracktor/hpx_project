cmake_minimum_required(VERSION 3.21)

project(GPRatTests LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
  find_package(GPRat REQUIRED)
  enable_testing()
endif()

# Option for GPU support with CUDA, cuSolver, cuBLAS
option(GPRAT_WITH_CUDA "Enable GPU support with CUDA, cuSolver, cuBLAS" OFF)
option(GPRAT_WITH_SYCL "Enable SYCL support with oneMath" OFF)

# Pass variable to C++ code
add_compile_definitions(GPRAT_WITH_CUDA=$<BOOL:${GPRAT_WITH_CUDA}>
                        GPRAT_WITH_SYCL=$<BOOL:${GPRAT_WITH_SYCL}>)

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

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.0)

FetchContent_MakeAvailable(Catch2)

# catch_discover_tests registers each Catch2 test case as a separate CTest
# entry.
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

find_package(Boost REQUIRED)

# ---- Tests ----

add_executable(GPRat_test_output_correctness src/test_data.hpp
                                             src/output_correctness.cpp)
target_link_libraries(GPRat_test_output_correctness
                      PRIVATE GPRat::core Catch2::Catch2WithMain Boost::boost)
target_compile_features(GPRat_test_output_correctness PRIVATE cxx_std_17)
# APEX's malloc wrapper initialises internal containers during process startup.
# With _GLIBCXX_ASSERTIONS enabled those accesses trip bounds checks before APEX
# is fully set up.  Undefine the assertion macro for this target only.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(GPRat_test_output_correctness
                         PRIVATE -U_GLIBCXX_ASSERTIONS)
endif()
catch_discover_tests(GPRat_test_output_correctness WORKING_DIRECTORY
                     "${CMAKE_CURRENT_LIST_DIR}")

add_executable(GPRat_test_unit src/unit_tests.cpp)
target_link_libraries(GPRat_test_unit PRIVATE GPRat::core
                                              Catch2::Catch2WithMain)
target_compile_features(GPRat_test_unit PRIVATE cxx_std_17)
catch_discover_tests(GPRat_test_unit WORKING_DIRECTORY
                     "${CMAKE_CURRENT_LIST_DIR}")

# Multi-locality smoke tests for examples/gprat_distributed. Off by default:
# they need an HPX build with networking enabled (networking=none, used by the
# default gprat_cpu_gcc Spack environment, rejects --hpx:localities outright) --
# see spack-repo/environments/setup_gprat_cpu_gcc_dist.sh and the "To run the
# distributed GPRat benchmark" section of the top-level README for how to build
# one.
option(
  GPRAT_TEST_MULTI_LOCALITY
  "Enable gprat_distributed smoke tests across multiple HPX localities (requires an HPX build with networking enabled)"
  OFF)
if(GPRAT_TEST_MULTI_LOCALITY)
  if(NOT TARGET gprat_distributed)
    message(
      FATAL_ERROR
        "GPRAT_TEST_MULTI_LOCALITY requires GPRAT_WITH_DISTRIBUTED and GPRAT_ENABLE_EXAMPLES to be ON"
    )
  endif()
  foreach(n_localities 1 2 4)
    add_test(
      NAME GPRat_test_distributed_multi_locality_${n_localities}
      COMMAND
        bash
        "${CMAKE_CURRENT_LIST_DIR}/scripts/run_distributed_multi_locality.sh"
        "$<TARGET_FILE:gprat_distributed>" ${n_localities} --start 128 --end
        128 --step 2 --tiles 2 --loop 1 --output_csv /dev/null)
    set_tests_properties(
      GPRat_test_distributed_multi_locality_${n_localities}
      PROPERTIES TIMEOUT
                 120
                 # The binary resolves its default data/data_1024/... paths
                 # relative to the current working directory, matching
                 # run_gprat_distributed.sh's `cd "$GPRAT_ROOT"` before invoking
                 # it.
                 WORKING_DIRECTORY
                 "${CMAKE_SOURCE_DIR}"
                 # Node 0 of each run binds HPX's default AGAS/parcel TCP port
                 # on localhost; running two of these tests concurrently makes
                 # the second's bind fail ("Address already in use"), which then
                 # hangs its orphaned peer processes until they hit TIMEOUT.
                 # Force them to run one at a time.
                 RUN_SERIAL
                 TRUE)
  endforeach()

  # gprat::GP (used by GPRat_test_output_correctness) always uses
  # tiled_scheduler_local, which is locality-oblivious -- it never exercises the
  # distributed action dispatch no matter how many localities are running. This
  # binary instead calls the same free functions examples/gprat_distributed uses
  # with tiled_scheduler_sma, so running it across multiple localities actually
  # verifies that dispatch, against the same data_1024/output.json baseline.
  add_executable(GPRat_test_distributed_output_correctness
                 src/test_data.hpp src/distributed_output_correctness.cpp)
  target_link_libraries(GPRat_test_distributed_output_correctness
                        PRIVATE GPRat::core Boost::boost)
  target_compile_features(GPRat_test_distributed_output_correctness
                          PRIVATE cxx_std_17)
  # Baked-in default data/ directory, overridable via the GPRAT_ROOT env var
  # (see test_data.hpp's get_data_directory()) so this always agrees with
  # GPRat_test_output_correctness on which data_1024/output.json baseline to
  # read/write.
  target_compile_definitions(
    GPRat_test_distributed_output_correctness
    PRIVATE GPRAT_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

  foreach(n_localities 1 2 4)
    add_test(
      NAME GPRat_test_distributed_output_correctness_${n_localities}
      COMMAND
        bash
        "${CMAKE_CURRENT_LIST_DIR}/scripts/run_distributed_multi_locality.sh"
        "$<TARGET_FILE:GPRat_test_distributed_output_correctness>"
        ${n_localities})
    set_tests_properties(
      GPRat_test_distributed_output_correctness_${n_localities}
      PROPERTIES TIMEOUT 120 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" RUN_SERIAL
                 TRUE)
  endforeach()
endif()
