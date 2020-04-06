
configure_file(config.pxi.in config.pxi)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CYTHON_EXECUTABLE "cython3")
include(UseCython)
find_package(Numpy REQUIRED )

include_directories(
   ${PYTHON_NUMPY_INCLUDE_DIR}
   .
   c
 )
add_subdirectory(c)
add_subdirectory(device)

set_source_files_properties(wrappers.c PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} ${CYTHON_EXTRA_C_FLAGS}")
cython_add_module(wrappers wrappers.pyx)
target_link_libraries(wrappers lasp_lib python37)
