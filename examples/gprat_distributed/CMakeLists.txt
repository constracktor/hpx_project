add_executable(gprat_distributed src/main.cpp)
# HPX_REGISTER_STARTUP_MODULE creates a static that icpx sees as unused
set_source_files_properties(src/main.cpp PROPERTIES COMPILE_OPTIONS
                                                    "-Wno-unused-variable")
target_compile_features(gprat_distributed PUBLIC cxx_std_20)

# Bake in the absolute path to this example directory so the default output.csv
# location works regardless of the current working directory, matching
# gprat_cpp.
target_compile_definitions(
  gprat_distributed PRIVATE GPRAT_DISTRIBUTED_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(gprat_distributed PUBLIC GPRat::core HPX::hpx)

set_target_properties(gprat_distributed PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY
                                                   "${CMAKE_SOURCE_DIR}")
