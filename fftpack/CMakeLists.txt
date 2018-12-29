# We borrow Numpy's implementation for doing the Fast Fourier Transform.
# This FFT code appears to be faster than KISSFFT.
add_library(fftpack
  fftpack.c
)
# Ling fft to math
target_link_libraries(fftpack m)
