#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description: Decimation filter design.
"""
import numpy as np
import matplotlib.pyplot as plt
from fir_design import freqResponse, lowpass_fir_design

L = 128  # Filter order
fs = 48000.  # Sampling frequency

d = 4 # Decimation factor

fd = fs/d  # Decimated sampling frequency
fc = fd/2/1.5  # Filter cut-off frequency

fir = lowpass_fir_design(L, fs, fc)

fig = plt.figure()
ax = fig.add_subplot(111)

freq = np.logspace(1, np.log10(fs/2), 1000)

H = freqResponse(fs, freq, fir)
dBH = 20*np.log10(np.abs(H))
ax.plot(freq, dBH)
ax.axvline(fs/2, color='green')
ax.axvline(fd/2, color='red')

# Index of Nyquist freq
fn_index = np.where(freq <= fd/2)[0][-1]

dBHmax_above_Nyq = np.max(dBH[fn_index:])

print(f"Reduction above Nyquist: {dBHmax_above_Nyq} dB")
plt.show()
