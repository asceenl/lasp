set_source_files_properties(lasp_daqdevice.c PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} ${CYTHON_EXTRA_C_FLAGS}")
# set_source_files_properties(lasp_portaudio.c PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} ${CYTHON_EXTRA_C_FLAGS}")
include_directories(/usr/include/rtaudio)
set_source_files_properties(lasp_rtaudio.pyx PROPERTIES CYTHON_IS_CXX TRUE)
set_source_files_properties(lasp_rtaudio.cxx PROPERTIES COMPILE_FLAGS
    "${CMAKE_CXX_FLAGS} ${CYTHON_EXTRA_CXX_FLAGS}")
cython_add_module(lasp_daqdevice lasp_daqdevice.pyx)
# cython_add_module(lasp_portaudio lasp_portaudio.pyx) */
cython_add_module(lasp_rtaudio lasp_rtaudio.pyx)

# target_link_libraries(lasp_portaudio portaudio) */
target_link_libraries(lasp_rtaudio rtaudio)
target_link_libraries(lasp_daqdevice asound)
