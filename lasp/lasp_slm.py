#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound level meter implementation
@author: J.A. de Jong - ASCEE
"""
from .wrappers import SPLowpass
import numpy as np
from .lasp_common import (TimeWeighting, P_REF)

__all__ = ['SLM', 'Dummy']


class Dummy:
    """
    Emulate filtering, but does not filter anything at all.
    """

    def filter_(self, data):
        return data[:, np.newaxis]


class SLM:
    """
    Multi-channel Sound Level Meter. Input data: time data with a certain
    sampling frequency. Output: time-weighted (fast/slow) sound pressure
    levels in dB(A/C/Z).

    """

    def __init__(self, fs, tw=TimeWeighting.default):
        """
        Initialize a sound level meter object.
        Number of channels comes from the weighcal object.

        Args:
            fs: Sampling frequency [Hz]
            tw: Time Weighting to apply
        """

        if tw[0] is not TimeWeighting.none[0]:
            # Initialize the single-pole low-pass filter for given time-
            # weighting value.
            self._lp = SPLowpass(fs, tw[0])
        else:
            self._lp = Dummy()
        self._Lmax = 0.

        # Storage for computing the equivalent level
        self._sq = 0. # Square of the level data, storage
        self._N = 0
        self._Leq = 0.

    @property
    def Leq(self):
        """
        Returns the equivalent level of the recorded signal so far
        """
        return self._Leq

    @property
    def Lmax(self):
        """
        Returns the currently maximum recorded level
        """
        return self._Lmax

    def addData(self, data):
        """
        Add new fresh timedata to the Sound Level Meter

        Args:
            data:
        returns:
            level values as a function of time
        """
        assert data.ndim == 2
        assert data.shape[1] == 1

        P_REFsq = P_REF**2

        # Squared
        sq = data**2

        # Update equivalent level
        N1 = sq.shape[0]
        self._sq = (np.sum(sq) + self._sq*self._N)/(N1+self._N)
        self._N += N1
        self._Leq = 10*np.log10(self._sq/P_REFsq)

        # Time-weight the signal
        tw = self._lp.filter_(sq)

        Level = 10*np.log10(tw/P_REFsq)

        # Update maximum level
        curmax = np.max(Level)
        if curmax > self._Lmax:
            self._Lmax = curmax

        return Level
