add_subdirectory(c)

configure_file(config.pxi.in config.pxi)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package( PythonLibs REQUIRED )
include(UseCython)


include_directories(
  .
  c
  )


set_source_files_properties(fft.c PROPERTIES COMPILE_FLAGS
  "-Wno-cpp -Wno-sign-compare -Wno-incompatible-pointer-types -Wno-strict-aliasing -Wno-implicit-fallthrough")
cython_add_module(fft fft.pyx)
target_link_libraries(fft beamforming_lib )
