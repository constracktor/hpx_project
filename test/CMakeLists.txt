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

find_package(Boost REQUIRED)

# ---- Tests ----

add_executable(GPRat_test_output_correctness src/output_correctness.cpp)
target_link_libraries(GPRat_test_output_correctness
                      PRIVATE GPRat::core Catch2::Catch2WithMain Boost::boost)
target_compile_features(GPRat_test_output_correctness PRIVATE cxx_std_17)

add_test(
  NAME GPRat_test_output_correctness
  COMMAND GPRat_test_output_correctness
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_executable(GPRat_test_unit src/unit_tests.cpp)
target_link_libraries(GPRat_test_unit PRIVATE GPRat::core
                                              Catch2::Catch2WithMain)
target_compile_features(GPRat_test_unit PRIVATE cxx_std_17)

add_test(
  NAME GPRat_test_unit
  COMMAND GPRat_test_unit
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")
