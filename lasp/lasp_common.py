#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from .wrappers import Window as wWindow
import appdirs, os, shelve
"""
Common definitions used throughout the code.
"""

__all__ = ['P_REF', 'FreqWeighting', 'TimeWeighting', 'getTime', 
           'getFreq', 'lasp_shelve',
           'W_REF', 'U_REF', 'I_REF']

lasp_appdir = appdirs.user_data_dir('Lasp', 'ASCEE')


if not os.path.exists(lasp_appdir):
    try:
        os.mkdir(lasp_appdir)
    except:
        print('Fatal error: could not create application directory')
        exit(1)


class lasp_shelve:
    refcount = 0
    shelve = None

    def __enter__(self):
        if lasp_shelve.shelve is None:
            assert lasp_shelve.refcount == 0
            lasp_shelve.shelve = shelve.open(os.path.join(lasp_appdir, 'config.shelve'))
        lasp_shelve.refcount += 1
        return lasp_shelve.shelve

    def __exit__(self, type, value, traceback):
        lasp_shelve.refcount -= 1
        if lasp_shelve.refcount == 0:
            lasp_shelve.shelve.close()
            lasp_shelve.shelve = None

# Reference sound pressure level
P_REF = 2e-5

W_REF = 1e-12  # 1 picoWatt
I_REF = 1e-12  # 1 picoWatt/ m^2

# Reference velocity for sound velocity level
U_REF = 5e-8


class Window:
    hann = (wWindow.hann, 'Hann')
    hamming = (wWindow.hamming, 'Hamming')
    rectangular = (wWindow.rectangular, 'Rectangular')
    bartlett = (wWindow.bartlett, 'Bartlett')
    blackman = (wWindow.blackman, 'Blackman')

    types = (hann, hamming, rectangular, bartlett, blackman)
    default = 0

    @staticmethod
    def fillComboBox(cb):
        """
        Fill Windows to a combobox

        Args:
            cb: QComboBox to fill
        """
        cb.clear()
        for tw in Window.types:
            cb.addItem(tw[1], tw)
        cb.setCurrentIndex(Window.default)

    def getCurrent(cb):
        return Window.types[cb.currentIndex()]

class TimeWeighting:
    none = (None, 'Raw (no time weighting)')
    uufast = (1e-4, '0.1 ms')
    ufast = (30e-3, '30 ms')
    fast = (0.125, 'Fast (0.125 s)')
    slow = (1.0, 'Slow (1.0 s)')
    tens = (10, '10 s')
    infinite = (np.Inf, 'Infinite')
    types = (none, uufast, ufast, fast, slow, tens, infinite)
    default = 2

    @staticmethod
    def fillComboBox(cb):
        """
        Fill TimeWeightings to a combobox

        Args:
            cb: QComboBox to fill
        """
        cb.clear()
        for tw in TimeWeighting.types:
            cb.addItem(tw[1], tw)
        cb.setCurrentIndex(TimeWeighting.default)

    def getCurrent(cb):
        return TimeWeighting.types[cb.currentIndex()]

class FreqWeighting:
    """
    Frequency weighting types
    """
    A = ('A', 'A-weighting')
    C = ('C', 'C-weighting')
    Z = ('Z', 'Z-weighting')
    types = (A, C, Z)
    default = 0

    @staticmethod
    def fillComboBox(cb):
        """
        Fill FreqWeightings to a combobox

        Args:
            cb: QComboBox to fill
        """
        cb.clear()
        for fw in FreqWeighting.types:
            cb.addItem(fw[1], fw)
        cb.setCurrentIndex(FreqWeighting.default)

    def getCurrent(cb):
        return FreqWeighting.types[cb.currentIndex()]

def getTime(fs, N, start=0):
    """
    Return a time array for given number of points and sampling frequency.

    Args:
        fs: Sampling frequency [Hz]
        N: Number of time samples
        start: Optional start ofset in number of samples
    """
    assert N > 0 and fs > 0
    return np.linspace(start, start + N/fs, N, endpoint=False)

def getFreq(fs, nfft):
    """
    return an array of frequencies for single-sided spectra

    Args:
        fs: Sampling frequency [Hz]
        nfft: Fft length (int)
    """
    df = fs/nfft   # frequency resolution
    K = nfft//2+1  # number of frequency bins
    return np.linspace(0, (K-1)*df, K)
