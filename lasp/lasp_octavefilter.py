#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Provides the FIR implementation of the octave filter bank

"""
__all__ = ['FirOctaveFilterBank', 'FirThirdOctaveFilterBank']

from .filter.filterbank_design import OctaveBankDesigner, ThirdOctaveBankDesigner
from .wrappers import Decimator, FilterBank as pyxFilterBank
import numpy as np


class FirFilterBank:
    """
    Single channel octave filter bank implementation
    """

    def __init__(self, fs):
        """
        Initialize a OctaveFilterBank object.

        Args:
            fs: Sampling frequency of base signal

        """
        assert np.isclose(fs, 48000), "Only sampling frequency" \
            " available is 48 kHz"

        maxdecimation = self.decimation(self.xs[0])
        self.decimators = []
        for dec in maxdecimation:
            self.decimators.append(Decimator(1, dec))

        xs_d1 = []
        xs_d4 = []
        xs_d16 = []
        xs_d64 = []
        xs_d256 = []

        self.filterbanks = []
        # Sort the x values in categories according to the required decimation
        for x in self.xs:
            dec = self.decimation(x)
            if len(dec) == 1 and dec[0] == 1:
                xs_d1.append(x)
            elif len(dec) == 1 and dec[0] == 4:
                xs_d4.append(x)
            elif len(dec) == 2:
                xs_d16.append(x)
            elif len(dec) == 3:
                xs_d64.append(x)
            elif len(dec) == 4:
                xs_d256.append(x)
            else:
                raise ValueError(f'No decimation found for x={x}')

        xs_all = [xs_d1, xs_d4, xs_d16, xs_d64, xs_d256]
        for xs in xs_all:
            nominals_txt = []
            firs = np.empty((self.L, len(xs)), order='F')
            for i, x in enumerate(xs):
                #  These are the filters that do not require lasp_decimation
                #  prior to filtering
                nominals_txt.append(self.nominal_txt(x))
                firs[:, i] = self.createFirFilter(fs, x)
            filterbank = {'fb': pyxFilterBank(firs, 1024),
                          'xs': xs,
                          'nominals': nominals_txt}
            self.filterbanks.append(filterbank)

        # Sample input counter.
        self.N = 0

        # Filter output counters
        self.dec = [1, 4, 16, 64, 256]

        # These intial delays are found experimentally using a toneburst
        # response.
        self.Nf = [915, 806, 780, 582, 338]

    def filterd(self, dec_stage, data):
        """
        Filter data for a given decimation stage

        Args:
            dec_stage: decimation stage
            data: Pre-filtered data
        """
        output = {}
        if data.shape[0] == 0:
            return output

        filtered = self.filterbanks[dec_stage]['fb'].filter_(data)
        Nf = filtered.shape[0]
        if Nf > 0:
            dec = self.dec[dec_stage]
            fd = self.fs/dec

            oldNf = self.Nf[dec_stage]
            tstart = oldNf/fd
            tend = tstart + Nf/fd

            t = np.linspace(tstart, tend, Nf, endpoint=False)
            self.Nf[dec_stage] += Nf
            for i, nom in enumerate(self.filterbanks[dec_stage]['nominals']):
                output[nom] = {'t': t, 'data': filtered[:, [i]]}

        return output

    def filter_(self, data):
        """
        Filter input data
        """
        assert data.ndim == 2
        assert data.shape[1] == 1, "invalid number of channels, should be 1"

        if data.shape[0] == 0:
            return {}

        # Output given as a dictionary with x as the key
        output = {}
        self.N += data.shape[0]

        output = {**output, **self.filterd(0, data)}

        for i in range(len(self.decimators)):
            dec_stage = i+1
            if data.shape[0] > 0:
                # Apply a decimation stage
                data = self.decimators[i].decimate(data)
                output = {**output, **self.filterd(dec_stage, data)}

        return output


class FirOctaveFilterBank(FirFilterBank, OctaveBankDesigner):
    """
    Filter bank which uses FIR filtering for each octave frequency band
    """

    def __init__(self, fs):
        OctaveBankDesigner.__init__(self)
        FirFilterBank.__init__(self, fs)


class FirThirdOctaveFilterBank(FirFilterBank, ThirdOctaveBankDesigner):
    """
    Filter bank which uses FIR filtering for each one-third octave frequency
    band.
    """

    def __init__(self, fs):
        ThirdOctaveBankDesigner.__init__(self)
        FirFilterBank.__init__(self, fs)
