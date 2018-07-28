set_source_files_properties(lasp_daqdevice.c PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} ${CYTHON_EXTRA_C_FLAGS}")
cython_add_module(lasp_daqdevice lasp_daqdevice.pyx)

target_link_libraries(lasp_daqdevice asound)
