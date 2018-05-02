# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
from .wrappers import Window as wWindow
"""
Common definitions used throughout the code.
"""

__all__ = ['P_REF', 'FreqWeighting', 'TimeWeighting', 'getTime', 'calfile',
           'sens']

# Reference sound pressure level
P_REF = 2e-5

# Todo: fix This
calfile = '/home/anne/wip/UMIK-1/cal/7027430_90deg.txt'
sens = 0.053690387255872614


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
    fast = (0.125, 'Fast')
    slow = (1.0, 'Slow')
    types = (none, uufast, ufast, fast, slow)
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

    return np.linspace(start/fs, N/fs, N, endpoint=False)


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
