set(fftpack_src numpy/fftpack.c)

add_library(fftpack
  fftpack.c
  )


target_link_libraries(fftpack m)
