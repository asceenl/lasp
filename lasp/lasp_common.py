#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import platform
import shelve

import appdirs
import numpy as np

from .wrappers import Window as wWindow

"""
Common definitions used throughout the code.
"""

__all__ = [
    'P_REF', 'FreqWeighting', 'TimeWeighting', 'getTime', 'getFreq',
    'lasp_shelve', 'this_lasp_shelve', 'W_REF', 'U_REF', 'I_REF'
]

lasp_appdir = appdirs.user_data_dir('Lasp', 'ASCEE')

if not os.path.exists(lasp_appdir):
    try:
        os.mkdir(lasp_appdir)
    except:
        print('Fatal error: could not create application directory')
        exit(1)

class Shelve:
    def load(self, key, default_value):
        """
        Load data from a given key, if key is not found, returns the
        default value if key is not found
        """
        if key in self.shelve.keys():
            return self.shelve[key]
        else:
            return default_value

    def __enter__(self):
        self.incref()
        return self

    def store(self, key, val):
        self._shelve[key] = val

    def deleteIfPresent(self, key):
        try:
            del self._shelve[key]
        except:
            pass

    def printAllKeys(self):
        print(list(self.shelve.keys()))

    def incref(self):
        if self.shelve is None:
            assert self.refcount == 0
            self.shelve = shelve.open(self.shelve_fn())
        self.refcount += 1

    def decref(self):
        self.refcount -= 1
        if self.refcount == 0:
            self.shelve.close()
            self.shelve = None

    def __exit__(self, type, value, traceback):
        self.decref()

class lasp_shelve(Shelve):
    _refcount = 0
    _shelve = None


    @property
    def refcount(self):
        return lasp_shelve._refcount

    @refcount.setter
    def refcount(self, val):
        lasp_shelve._refcount = val

    @property
    def shelve(self):
        return lasp_shelve._shelve

    @shelve.setter
    def shelve(self, shelve):
        lasp_shelve._shelve = shelve

    def shelve_fn(self):
        return os.path.join(lasp_appdir, 'config.shelve')

class this_lasp_shelve(Shelve):
    _refcount = 0
    _shelve = None

    @property
    def refcount(self):
        return this_lasp_shelve._refcount

    @refcount.setter
    def refcount(self, val):
        this_lasp_shelve._refcount = val

    @property
    def shelve(self):
        return this_lasp_shelve._shelve

    @shelve.setter
    def shelve(self, shelve):
        this_lasp_shelve._shelve = shelve

    def shelve_fn(self):
        node = platform.node()
        return os.path.join(lasp_appdir, f'{node}_config.shelve')


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
    none = (-1, 'Raw (no time weighting)')
    uufast = (1e-4, '0.1 ms')
    ufast = (35e-3, 'Impulse (35 ms)')
    fast = (0.125, 'Fast (0.125 s)')
    slow = (1.0, 'Slow (1.0 s)')
    tens = (10., '10 s')
    infinite = (0, 'Infinite')
    types = (none, uufast, ufast, fast, slow, tens)
    types_all = (none, uufast, ufast, fast, slow, tens, infinite)
    default = fast
    default_index = 3

    @staticmethod
    def fillComboBox(cb, all_=False):
        """
        Fill TimeWeightings to a combobox

        Args:
            cb: QComboBox to fill
        """
        cb.clear()
        if all_:
            types = TimeWeighting.types_all
        else:
            types = TimeWeighting.types
        for tw in types:
            cb.addItem(tw[1], tw)
        cb.setCurrentIndex(TimeWeighting.default_index)

    @staticmethod
    def getCurrent(cb):
        return TimeWeighting.types_all[cb.currentIndex()]

class FreqWeighting:
    """
    Frequency weighting types
    """
    A = ('A', 'A-weighting')
    C = ('C', 'C-weighting')
    Z = ('Z', 'Z-weighting')
    types = (A, C, Z)
    default = A
    default_index = 0

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
        cb.setCurrentIndex(FreqWeighting.default_index)

    @staticmethod
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
