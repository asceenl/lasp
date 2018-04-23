#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighting and calibration filter in one
@author: J.A. de Jong - ASCEE
"""
from .filter_design.freqweighting_fir import A,C
from .lasp_common import FreqWeighting
from .filter_design.fir_design import (arbitrary_fir_design,
                                       freqResponse as frp)
from lasp.lasp_config import ones, empty
from .wrappers import FilterBank
import numpy as np

__all__ = ['WeighCal']


class WeighCal:
    """
    Time weighting and calibration FIR filter
    """
    def __init__(self, fw = FreqWeighting.default,
                 nchannels = 1,
                 fs = 48000.,
                 calfile = None,
                 sens=1.0):
        """
        Initialize the frequency weighting and calibration FIR filters.

        Args:
            fw: Frequency weighting to apply
            nchannels: Number of channels for the input data
            fs: Sampling frequency [Hz]
            calfile: Calibration file to load.
            sens: Sensitivity in units [\f$ Pa^{-1} \f$]
        """

        self.nchannels = nchannels
        self.fs = fs
        self.fw = fw
        self.sens = sens
        self.calfile = calfile

        # Frequencies used for the filter design
        freq_design = np.linspace(0,17e3,3000)
        freq_design[-1] = fs/2

        # Objective function for the frequency response
        frp_obj = self.frpObj(freq_design)

        self._firs = []
        self._fbs = []
        for chan in range(self.nchannels):
            fir  = arbitrary_fir_design(fs,2048,freq_design,
                                                  frp_obj[:,chan],
                                                  window='rectangular')
            self._firs.append(fir)

            self._fbs.append(FilterBank(fir[:,np.newaxis],4096))

        self._freq_design = freq_design

    def filter_(self,data):
        """
        Filter data using the calibration and frequency weighting filter.
        """
        nchan = self.nchannels
        assert data.shape[1] == nchan
        assert data.shape[0] > 0

        filtered = []
        for chan in range(nchan):
            filtered.append(self._fbs[chan].filter_(data[:,chan])[:,0])
        filtered = np.asarray(filtered).transpose()/self.sens
        if filtered.ndim == 1:
            filtered = filtered[:,np.newaxis]
        return filtered


    def frpCalObj(self, freq_design):
        """
        Computes the objective frequency response of the calibration filter
        """
        calfile = self.calfile
        if calfile is not None:
            cal = np.loadtxt(calfile,skiprows=2)
            freq = cal[:,0]
            cal = cal[:,1:]
            if cal.shape[1] != self.nchannels:
                raise ValueError('Number of channels in calibration file does'
                                 ' not equal to given number of channels')
            calfac = 10**(-cal/20)
            filter_calfac = empty((freq_design.shape[0],self.nchannels))

            for chan in range(self.nchannels):
                filter_calfac[:,chan] = np.interp(freq_design,freq,calfac[:,chan])

        else:
            filter_calfac = ones((freq_design.shape[0],self.nchannels,))

        return filter_calfac

    def frpWeightingObj(self, freq_design):
        """
        Computes the objective frequency response of the frequency weighting
        filter.
        """
        fw = self.fw
        if fw == FreqWeighting.A:
            return A(freq_design)
        elif fw == FreqWeighting.C:
            return C(freq_design)
        elif fw == FreqWeighting.Z:
            return ones(freq_design.shape[0])
        else:
            raise ValueError('Invalid fw parameter')

    def frpObj(self, freq_design):
        """
        Combines the frequency weighting and the calibration filter into
        one frequency response objective function.
        """
        # Objective function for the frequency response
        frp_objective = self.frpCalObj(freq_design) * \
                        self.frpWeightingObj(freq_design)[:,np.newaxis]
        frp_objective[-1] = 0.

        return frp_objective

    def freqResponse(self, chan=0, freq=None):
        """
        Returns the frequency response of the designed FIR filter
        """
        if freq is None:
            freq = np.logspace(1,np.log10(self.fs/2),500)
        return freq,frp(self.fs,freq,self._firs[chan]),self.frpObj(freq)[:,chan]
