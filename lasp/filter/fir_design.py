#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description: Designs octave band FIR filters from 16Hz to 16 kHz for a sampling
frequency of 48 kHz.
"""
# from asceefigs.plot import Bode, close, Figure
__all__ = ['freqResponse', 'bandpass_fir_design', 'lowpass_fir_design',
           'arbitrary_fir_design']

import numpy as np
from scipy.signal import freqz, hann, firwin2


def freqResponse(fs, freq, coefs_b, coefs_a=1.):
    """
    Computes the frequency response of the filter defined with the filter
    coefficients:

    Args:
        fs: Sampling frequency [Hz]
        freq: Array of frequencies to compute the response for
        coefs_b: Forward coefficients (FIR coefficients)
        coefs_a: Feedback coefficients (IIR)

    Returns:
        Complex frequency response for frequencies given in array
    """
    Omg = 2*np.pi*freq/fs

    w, H = freqz(coefs_b, coefs_a, worN=Omg)
    return H


def bandpass_fir_design(L, fs, fl, fu, window=hann):
    """
    Construct a bandpass filter
    """
    assert fs/2 > fu, "Nyquist frequency needs to be higher than upper cut-off"
    assert fu > fl, "Cut-off needs to be lower than Nyquist freq"

    Omg2 = 2*np.pi*fu/fs
    Omg1 = 2*np.pi*fl/fs

    fir = np.empty(L, dtype=float)

    # First Create ideal band-pass filter
    fir[L//2] = (Omg2-Omg1)/np.pi

    for n in range(1, L//2):
        fir[n+L//2] = (np.sin(n*Omg2)-np.sin(n*Omg1))/(n*np.pi)
        fir[L//2-n] = (np.sin(n*Omg2)-np.sin(n*Omg1))/(n*np.pi)

    win = window(L, True)
    fir_win = fir*win

    return fir_win


def lowpass_fir_design(L, fs, fc, window=hann):
    assert fs/2 > fc, "Nyquist frequency needs to be higher" \
                      " than upper cut-off"

    Omgc = 2*np.pi*fc/fs
    fir = np.empty(L, dtype=float)

    # First Create ideal band-pass filter
    fir[L//2] = Omgc/np.pi

    for n in range(1, L//2):
        fir[n+L//2] = np.sin(n*Omgc)/(n*np.pi)
        fir[L//2-n] = np.sin(n*Omgc)/(n*np.pi)

    win = window(L, True)
    fir_win = fir*win

    return fir_win


def arbitrary_fir_design(fs, L, freq, amps, window='hann'):
    """
    Last frequency of freq should be fs/2
    """
    return firwin2(L, freq, amps, fs=fs, window=window)
