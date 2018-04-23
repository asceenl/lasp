#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound level meter implementation
@author: J.A. de Jong - ASCEE
"""
from .wrappers import FilterBank, SPLowpass
import numpy as np
from .lasp_config import zeros
from .lasp_common import TimeWeighting, P_REF
__all__ = ['SLM']

class Dummy:
    def filter_(self, data):
        return data

class SLM:
    """
    Sound Level Meter, implements the single pole lowpass filter
    """
    def __init__(self, fs, weighcal,
                 tw=TimeWeighting.default,
                 nchannels = 1,
                 freq=None, cal=None):

        self._weighcal = weighcal
        if tw is not TimeWeighting.none:
            self._lps = [SPLowpass(fs, tw[0]) for i in range(nchannels)]
        else:
            self._lpw = [Dummy() for i in range(nchannels)]
        self._Lmax = zeros(nchannels)

    @property
    def Lmax(self):
        return self._Lmax

    def addData(self, data):
        assert data.ndim == 2
        data_weighted = self._weighcal.filter_(data)

        # Squared
        sq = data_weighted**2
        if sq.shape[0] == 0:
            return np.array([])

        tw = []
        # Time-weight the signal
        for chan,lp in enumerate(self._lps):
            tw.append(lp.filter_(sq[:,chan])[:,0])
        tw = np.asarray(tw).transpose()

        Level = 10*np.log10(tw/P_REF**2)

        curmax = np.max(Level)
        if curmax > self._Lmax:
            self._Lmax = curmax

        return Level
