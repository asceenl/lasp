#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

"""
import numpy as np
from lasp.filter.bandpass_limits import (third_octave_band_limits,
                                         octave_band_limits, G, fr)

from lasp.filter.bandpass_fir import ThirdOctaveBankDesigner, \
                                     OctaveBankDesigner
import matplotlib.pyplot as plt

# Adjust these settings
b = 1  # or three
zoom = False  # or True

if b == 3:
    bands = ThirdOctaveBankDesigner()
elif b == 1:
    bands = OctaveBankDesigner()
else:
    raise ValueError('b should be 1 or 3')


for x in bands.xs:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fs = 48000.
    dec = np.prod(bands.decimation(x))
    fd = fs/dec
    fc = fd/2/1.4

    freq = np.logspace(np.log10(1), np.log10(fd), 5000)
    H = bands.freqResponse(fs, x, freq)
    dBH = 20*np.log10(np.abs(H))
    ax.semilogx(freq, dBH)

    if b == 1:
        freq, ulim, llim = octave_band_limits(x)
    else:
        freq, ulim, llim = third_octave_band_limits(x)

    ax.semilogx(freq, llim)
    ax.semilogx(freq, ulim)
    ax.set_title(f'x = {x}, fnom = {bands.nominal_txt(x)}')

    if zoom:
        ax.set_xlim(bands.fl(x)/1.1, bands.fu(x)*1.1)
        ax.set_ylim(-15, 1)
    else:
        ax.set_ylim(-75, 1)
        ax.set_xlim(10, fd)

    ax.axvline(fd/2)
    if dec > 1:
        ax.axvline(fc, color='red')

    ax.legend(['Filter frequency response',
               'Lower limit from standard',
               'Upper limit from standard',
               'Nyquist frequency after decimation',
               'Decimation filter cut-off frequency'], fontsize=8)

plt.show()
