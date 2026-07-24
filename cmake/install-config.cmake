# Same as in the root CMakeLists.txt
if(GPRAT_ENABLE_MKL)
  # Try to find Intel oneMKL
  set(MKL_INTERFACE_FULL "intel_lp64")
  set(MKL_THREADING "sequential")
  find_package(MKL CONFIG REQUIRED)
else()
  # Try to find OpenBLAS
  find_library(OpenBLAS_LIB NAMES openblas REQUIRED)
endif()

find_package(HPX REQUIRED)
include("${CMAKE_CURRENT_LIST_DIR}/hpx-apex-compat.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/GPRatTargets.cmake")
