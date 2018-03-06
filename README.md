# Library for Acoustic Signal Processing

Welcome to LASP: Library for Acoustic Signal Processing. LASP is a C
library - currently still under heavy development - with a Python
interface which is supposed to process (multi-) microphone acoustic
data in real time and output results.

The main goal of this library will be the processing of data from an
array of microphones real time, on a Raspberry PI. At the point in
time of this writing, we are yet unsure whether the Raspberry PI will
have enough computational power to this end, but may be by the time it
is finished, we have a new faster generation :).

Current features that are implemented:
- Compile-time determination of the floating-point accuracy (32/64 bit)
- Fast convolution FIR filter implementation
- Decimation of the sample rate by an integer factor of 4.
- Octave filterbank FIR filters designed to be compliant to IEC 61260
  (1995).


Some of the near future features:
- Third octave filter bank
- Slow and fast time updates of (A/C/Z) weighted sound pressure levels
- Conventional and delay-and-sum beamforming algorithms

For now, the source code is well-documented but it requires some
additional documentation (the math behind it). This will be published
in a sister repository in a later stage.

If you have any question, please feel free to contact us: info@ascee.nl.

