#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description: Implementation of SOS filters to color a signal. Typically used to
color white noise to i.e. Pink noise, or Brownian noise.
"""
from scipy.signal import bilinear_zpk, zpk2sos, freqz_zpk
import numpy as np

__all__ = ['PinkNoise'] #, 'BrownianNoise', 'BlueNoise']

def PinkNoise(fs, fstart=10, fend=None, N=3):
    """
    Creates SOS filter for pink noise. The filter has a flat response below
    fstart, and rolls of close to Nyquist, or flattens out above fend. The
    ripple (i.e.) closeness the filter rolls of depends on the number of
    sections. The default, N=3 results in

    Args:
        fs: Sampling frequency [Hz]
        fstart: Frequency of first pole of the filter
        fend: Frequency of last pole of the filter, if not given, set to 5/6th
        of the Nyquist frequency.
        N: Number of sections.

    Returns:
        sos: Array of digital filter coefficients

    """
    order = N*2
    if fend is None:
        fend = 5*fs/6/2
    # Not the fastest implementation, but this will not be the bottleneck
    # anyhow.
    fpoles = np.array([fstart*(fend/fstart)**(n/(order-1)) 
                       for n in range(order)])

    # Put zeros inbetween poles
    fzeros = np.sqrt(fpoles[1:]*fpoles[:-1])

    poles = -2*np.pi*fpoles
    zeros = -2*np.pi*fzeros

    z,p,k = bilinear_zpk(zeros, poles,1, fs=fs)

    # Compute the frequency response and search for the gain at fstart, we
    # normalize such that this gain is ~ 0 dB. Rounded to an integer frequency
    # in Hz.
    Omg, h = freqz_zpk(z, p, k, worN = int(fs/2), fs=fs)
    h_fstart = np.abs(h[int(fstart)])
    k *= 1/h_fstart
    
    sos = zpk2sos(z,p,k)

    return sos
