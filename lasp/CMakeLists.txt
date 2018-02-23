add_subdirectory(c)

configure_file(config.pxi.in config.pxi)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package( PythonLibs REQUIRED )
include(UseCython)


include_directories(
  .
  c
  )


set_source_files_properties(wrappers.c PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} ${CYTHON_EXTRA_C_FLAGS}")  
cython_add_module(wrappers wrappers.pyx)
target_link_libraries(wrappers beamforming_lib )
