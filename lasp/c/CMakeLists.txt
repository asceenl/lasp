if(!LASP_DEBUG)
# SET_SOURCE_FILES_PROPERTIES(y.c PROPERTIES COMPILE_FLAGS -O3)
# SET_SOURCE_FILES_PROPERTIES(x.c PROPERTIES COMPILE_FLAGS -O3)
endif(!LASP_DEBUG)

add_library(lasp_lib
  lasp_fft.c
  lasp_math.c
  lasp_math_raw.c
  lasp_alg.c
  lasp_assert.c
  lasp_tracer.c
  lasp_window.c
  lasp_aps.c
  lasp_ps.c
  lasp_mq.c
  lasp_worker.c
  lasp_dfifo.c
  lasp_filterbank.c
  )


target_link_libraries(lasp_lib fftpack openblas)
