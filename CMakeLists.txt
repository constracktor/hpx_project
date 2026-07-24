# Top-level CMakeLists.txt

# CMake initialization
# ##############################################################################
cmake_minimum_required(VERSION 3.23)

project(
  GPRat
  VERSION 0.1.0
  DESCRIPTION "Gaussian Process Regression using Asynchronous Tasks"
  HOMEPAGE_URL "https://github.com/SC-SGS/GPRat"
  LANGUAGES CXX)

include(CMakeDependentOption)

# Build options
# ##############################################################################
option(GPRAT_BUILD_CORE "Build the core library" ON)

cmake_dependent_option(GPRAT_BUILD_BINDINGS "Build the Python bindings" ON
                       "GPRAT_BUILD_CORE" OFF)

cmake_dependent_option(
  GPRAT_ENABLE_EXAMPLES "Build example applications as well?"
  ${PROJECT_IS_TOP_LEVEL} "GPRAT_BUILD_CORE" OFF)

cmake_dependent_option(GPRAT_ENABLE_TESTS "Build unit and integration tests"
                       ${PROJECT_IS_TOP_LEVEL} "GPRAT_BUILD_CORE" OFF)

cmake_dependent_option(GPRAT_ENABLE_MKL "Enable support for Intel oneMKL" OFF
                       "GPRAT_BUILD_CORE" OFF)
option(GPRAT_ENABLE_BENCHMARK_CACHE_EVICTIONS
       "Evict data from caches before running BLAS operations" ON)

option(GPRAT_ENABLE_FORMAT_TARGETS "Enable clang-format / cmake-format targets"
       ${PROJECT_IS_TOP_LEVEL})

# Format settings
# ##############################################################################
if(GPRAT_ENABLE_FORMAT_TARGETS)
  set(CMAKE_FORMAT_EXCLUDE "^external_ports/")

  find_package(format QUIET)
  if(NOT format_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      format
      GIT_REPOSITORY https://github.com/TheLartians/Format.cmake.git
      GIT_TAG v1.8.1
      QUIET)
    FetchContent_MakeAvailable(format)
  endif()
endif()

# GNU install dirs
# ##############################################################################
if(NOT CMAKE_SKIP_INSTALL_RULES)
  # Our installs follow the standard GNU directory layout. This include needs to
  # come first since we need the CMAKE_INSTALL_* in the CMakeLists.txt of each
  # target.
  include(GNUInstallDirs)
endif()

# Building the GPRAT core
# ##############################################################################
if(GPRAT_BUILD_CORE)

  # MKL backend
  if(GPRAT_ENABLE_MKL)
    set(MKL_INTERFACE_FULL "intel_lp64")
    set(MKL_THREADING "sequential")
    find_package(MKL CONFIG REQUIRED)
    if(MKL_FOUND)
      message(STATUS "Intel oneMKL Library found")
    else()
      message(FATAL_ERROR "Intel oneMKL Library NOT found")
    endif()

    # OpenBLAS backend
  else()
    find_library(OpenBLAS_LIB NAMES openblas REQUIRED)
    if(OpenBLAS_LIB)
      message(STATUS "OpenBLAS Library found at ${OpenBLAS_LIB}")
    else()
      message(FATAL_ERROR "No BLAS Library found")
    endif()
  endif()

  # HPX
  find_package(HPX REQUIRED)
  include(cmake/hpx-apex-compat.cmake)

  # Add core subdirectiory
  add_subdirectory(core)

  # Bindings subdirectory
  if(GPRAT_BUILD_BINDINGS)
    add_subdirectory(bindings)
  endif()

endif()

# Installation
# ##############################################################################
if(NOT CMAKE_SKIP_INSTALL_RULES AND GPRAT_BUILD_CORE)
  include(CMakePackageConfigHelpers)

  # find_package(<package>) call for consumers to find this project
  set(package GPRat)

  write_basic_package_version_file("${package}ConfigVersion.cmake"
                                   COMPATIBILITY SameMajorVersion)

  # Allow package maintainers to freely override the path for the configs
  set(GPRat_INSTALL_CMAKEDIR
      "${CMAKE_INSTALL_LIBDIR}/cmake/${package}"
      CACHE STRING
            "CMake package config location relative to the install prefix")
  set_property(CACHE GPRat_INSTALL_CMAKEDIR PROPERTY TYPE PATH)
  mark_as_advanced(GPRat_INSTALL_CMAKEDIR)

  install(
    FILES cmake/install-config.cmake
    DESTINATION "${GPRat_INSTALL_CMAKEDIR}"
    RENAME "${package}Config.cmake"
    COMPONENT Development)

  install(
    FILES cmake/hpx-apex-compat.cmake
    DESTINATION "${GPRat_INSTALL_CMAKEDIR}"
    COMPONENT Development)

  install(
    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
    DESTINATION "${GPRat_INSTALL_CMAKEDIR}"
    COMPONENT Development)

  install(
    EXPORT GPRatTargets
    NAMESPACE GPRat::
    DESTINATION "${GPRat_INSTALL_CMAKEDIR}"
    COMPONENT Development)

  if(PROJECT_IS_TOP_LEVEL)
    include(CPack)
  endif()
endif()

# Examples
# ##############################################################################
if(GPRAT_ENABLE_EXAMPLES)
  add_subdirectory(examples/gprat_cpp)
  if(GPRAT_WITH_DISTRIBUTED)
    add_subdirectory(examples/gprat_distributed)
  endif()
endif()

# Tests
# ##############################################################################
if(GPRAT_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

# End of file
# ##############################################################################
