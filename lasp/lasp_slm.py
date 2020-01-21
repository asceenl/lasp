#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound level meter implementation
@author: J.A. de Jong - ASCEE
"""
from .wrappers import Slm as pyxSlm
import numpy as np
from .lasp_common import (TimeWeighting, P_REF, FreqWeighting)
from .filter import SPLFilterDesigner

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
    levels in dB(A/C/Z). Possibly in octave bands.

    """

    def __init__(self,
                 fbdesigner, 
                 tw=TimeWeighting.fast,
                 fw=FreqWeighting.A,
                 include_overall=True):
        """
        Initialize a sound level meter object.

        Args:
            fbdesigner: FilterBankDesigner to use for creating the 
            (fractional) octave bank filters
            fs: Sampling frequency [Hz]
            tw: Time Weighting to apply
            fw: Frequency weighting to apply
            include_overall: If true, a non-functioning filter is added which
            is used to compute the overall level.

            """

        self.fbdesigner = fbdesigner
        self.xs = fbdesigner.xs[:]

        nfilters = len(self.xs)
        if include_overall: nfilters +=1
        self.include_overall = include_overall
        
        fs = fbdesigner.fs
        spld = SPLFilterDesigner(fs)
        if fw == FreqWeighting.A:
            prefilter = spld.A_Sos_design().flatten()
        elif fw == FreqWeighting.C:
            prefilter = spld.C_Sos_design().flatten()
        elif fw == FreqWeighting.Z:
            prefilter = None
        else:
            raise ValueError('Not implemented prefilter')

        # 'Probe' size of filter coefficients
        self.nom_txt = []

        sos0 = fbdesigner.createSOSFilter(self.xs[0]).flatten()
        sos = np.empty((nfilters, sos0.size), dtype=float, order='C')
        sos[0, :] = sos0

        for i, x in enumerate(self.xs[1:]):
            sos[i, :] = fbdesigner.createSOSFilter(x).flatten()
            self.nom_txt.append(fbdesigner.nominal_txt(x))

        if include_overall:
            # Create a unit impulse response filter, every third index equals 
            # 1, so b0 = 1 and a0 is 1 (by definition)
            sos[-1,:] = 0
            sos[-1,::3] = 1
            self.nom_txt.append('overall')

        self.slm = pyxSlm(prefilter, sos,
                          fs, tw[0], P_REF)

        dsfac = self.slm.downsampling_fac
        if dsfac > 0:
            # Not unfiltered data
            self.fs_slm = fs / self.slm.downsampling_fac
        else:
            self.fs_slm = fs

        # Initialize counter to 0
        self.N = 0


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

    def run(self, data):
        """
        Add new fresh timedata to the Sound Level Meter

        Args:
            data: one-dimensional input data
        """

        assert data.ndim == 2
        assert data.shape[1] == 1, "invalid number of channels, should be 1"

        if data.shape[0] == 0:
            return {}

        levels = self.slm.run(data)

        tstart = self.N / self.fs_slm
        Ncur = levels.shape[0]
        tend = tstart + Ncur / self.fs_slm

        t = np.linspace(tstart, tend, Ncur, endpoint=False)
        self.N += Ncur

        output = {}

        for i, x in enumerate(self.xs):
            # '31.5' to '16k'
            output[self.nom_txt[i]] = {'t': t, 
                                       'data': levels[:, i],
                                       'x': x}
        if self.include_overall:
            output['overall'] = {'t': t, 'data': levels[:, i+1], 'x': 0}
        return output


