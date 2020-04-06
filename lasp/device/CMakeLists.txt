include_directories(/usr/include/rtaudio)
set_source_files_properties(lasp_rtaudio.pyx PROPERTIES CYTHON_IS_CXX TRUE)
set_source_files_properties(lasp_rtaudio.cxx PROPERTIES COMPILE_FLAGS
    "${CMAKE_CXX_FLAGS} ${CYTHON_EXTRA_CXX_FLAGS}")

cython_add_module(lasp_rtaudio lasp_rtaudio.pyx)

target_link_libraries(lasp_rtaudio pthread rtaudio python37)