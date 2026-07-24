# HPX built with +static and instrumentation=apex embeds APEX's private
# zlib/rapidjson/otf2 dependencies into HPXTargets.cmake by bare name instead of
# as proper (exported) targets. Since no target with those names exists in a
# consuming project, CMake falls back to raw "-l<name>" linker flags, which
# fail: "-lzlib" has no matching library file (real zlib produces libz, not
# libzlib) and "-lrapidjson" is header-only and never produces a library file at
# all. Defining targets with these exact names satisfies
# target_link_libraries()'s lookup before it degrades to a linker flag. This is
# purely additive: targets are only created when the real dependency can be
# found, so builds that don't hit this HPX export bug are unaffected.
if(NOT TARGET zlib)
  find_package(ZLIB QUIET)
  if(ZLIB_FOUND)
    add_library(zlib INTERFACE IMPORTED)
    target_link_libraries(zlib INTERFACE ZLIB::ZLIB)
  endif()
endif()

if(NOT TARGET rapidjson)
  add_library(rapidjson INTERFACE IMPORTED)
endif()

if(NOT TARGET otf2)
  find_library(
    GPRat_OTF2_LIBRARY
    NAMES otf2
    HINTS "${Otf2_ROOT}/lib")
  if(GPRat_OTF2_LIBRARY)
    add_library(otf2 INTERFACE IMPORTED)
    target_link_libraries(otf2 INTERFACE "${GPRat_OTF2_LIBRARY}")
  endif()
endif()
