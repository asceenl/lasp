if(!ASCEE_DEBUG)
# SET_SOURCE_FILES_PROPERTIES(si_lpn.c PROPERTIES COMPILE_FLAGS -O3)
# SET_SOURCE_FILES_PROPERTIES(si_math.c PROPERTIES COMPILE_FLAGS -O3)
endif(!ASCEE_DEBUG)

add_library(beamforming_lib
  fft.c
  ascee_math.c
  ascee_assert.c
  tracer.c
  window.c
  # ps.c
  mq.c
  worker.c
  )


target_link_libraries(beamforming_lib fftpack openblas)
