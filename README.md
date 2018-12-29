# Library for Acoustic Signal Processing

Welcome to LASP: Library for Acoustic Signal Processing. LASP is a C library
with a Python interface which is supposed to process (multi-) microphone
acoustic data in real time on a PC and output results.

The main goal of this library will be the processing of data from an
array of microphones real time, on a Raspberry PI. At the point in
time of this writing, we are yet unsure whether the Raspberry PI will
have enough computational power to this end, but may be by the time it
is finished, we have a new faster generation :).

Current features that are implemented:
- Compile-time determination of the floating-point accuracy (32/64 bit)
- Fast convolution FIR filter implementation
- Sample rate decimation by an integer factor of 4.
- Octave filterbank FIR filters designed to comply with IEC 61260
  (1995).
- Averaged power spectra and power spectral density determination
  using Welch' method. Taper functions of Hann, Hamming, Bartlett and
  Blackman are provided.
- A thread-safe job queue including routines to create worker threads.
- Several linear algebra routines (wrappers around BLAS and LAPACK).
- A nice debug tracer implementation
- Third octave filter bank FIR filters designed to comply with IEC 61260
  (1995).
- Slow and fast time updates of (A/C/Z) weighted sound pressure levels

Future features (wish-list)
- Conventional and delay-and-sum beam-forming algorithms

For now, the source code is well-documented but it requires some
additional documentation (the math behind it). This will be published
in a sister repository in a later stage.

If you have any question(s), please feel free to contact us: info@ascee.nl.
