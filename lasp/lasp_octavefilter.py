#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Provides the implementations of (fractional) octave filter banks

"""
__all__ = ['FirOctaveFilterBank', 'FirThirdOctaveFilterBank',
           'OverallFilterBank', 'SosOctaveFilterBank',
           'SosThirdOctaveFilterBank']

from .filter.filterbank_design import (OctaveBankDesigner,
                                       ThirdOctaveBankDesigner)
from .wrappers import (Decimator, FilterBank as pyxFilterBank,
                       SosFilterBank as pyxSosFilterBank)
import numpy as np


class OverallFilterBank:
    """
    Dummy type filter bank. Does nothing special, only returns output in a
    sensible way
    """

    def __init__(self, fs):
        """
        Initialize overall filter bank
        """
        self.fs = fs
        self.N = 0
        self.xs = [0]

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

        tstart = self.N / self.fs
        Ncur = data.shape[0]
        tend = tstart + Ncur / self.fs

        t = np.linspace(tstart, tend, Ncur, endpoint=False)
        self.N += Ncur

        output['Overall'] = {'t': t, 'data': data, 'x': 0}
        return output

    def decimation(self, x):
        return [1]


class FirFilterBank:
    """
    Single channel (fractional) octave filter bank implementation, based on FIR
    filters and sample rate decimation.
    """

    def __init__(self, fs, xmin, xmax):
        """
        Initialize a OctaveFilterBank object.

        Args:
            fs: Sampling frequency of base signal

        """
        assert np.isclose(fs, 48000), "Only sampling frequency" \
            " available is 48 kHz"

        self.fs  = fs
        self.xs = list(range(xmin, xmax + 1))

        maxdecimation = self.designer.firDecimation(self.xs[0])
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
            dec = self.designer.firDecimation(x)
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
            if len(xs) > 0:
                firs = np.empty((self.designer.firFilterLength, len(xs)), order='F')
                for i, x in enumerate(xs):
                    #  These are the filters that do not require lasp_decimation
                    #  prior to filtering
                    nominals_txt.append(self.designer.nominal_txt(x))
                    firs[:, i] = self.designer.createFirFilter(x)
                filterbank = {'fb': pyxFilterBank(firs, 1024),
                              'xs': xs,
                              'nominals': nominals_txt}
                self.filterbanks.append(filterbank)

        # Sample input counter.
        self.N = 0

        self.dec = [1, 4, 16, 64, 256]

        # Filter output counters
        # These intial delays are found 'experimentally' using a toneburst
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
            for i, nom_txt in enumerate(self.filterbanks[dec_stage]['nominals']):
                x = self.designer.nominal_txt_tox(nom_txt)
                output[nom_txt] = {'t': t, 'data': filtered[:, [i]], 'x': x}

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
        output_unsorted = {}
        self.N += data.shape[0]

        output_unsorted = {**output_unsorted, **self.filterd(0, data)}

        for i in range(len(self.decimators)):
            dec_stage = i+1
            if data.shape[0] > 0:
                # Apply a decimation stage
                data = self.decimators[i].decimate(data)
                output_unsorted = {**output_unsorted,
                                   **self.filterd(dec_stage, data)}

        # Create sorted output
        for x in self.xs:
            nom_txt = self.designer.nominal_txt(x)
            output[nom_txt] = output_unsorted[nom_txt]

        return output

    def decimation(self, x):
        return self.designer.firDecimation(x)


class FirOctaveFilterBank(FirFilterBank):
    """
    Filter bank which uses FIR filtering for each octave frequency band
    """

    def __init__(self, fs, xmin, xmax):
        self.designer = OctaveBankDesigner(fs)
        FirFilterBank.__init__(self, fs, xmin, xmax)


class FirThirdOctaveFilterBank(FirFilterBank):
    """
    Filter bank which uses FIR filtering for each one-third octave frequency
    band.
    """

    def __init__(self, fs, xmin, xmax):
        self.designer = ThirdOctaveBankDesigner(fs)
        FirFilterBank.__init__(self, fs, xmin, xmax)


class SosFilterBank:
    def __init__(self, fs, xmin, xmax):
        """
        Initialize a second order sections filterbank

        Args:
            fs: Sampling frequency [Hz]
            xmin: Minimum value for the bands
            xmax: Maximum value for the bands
        """
        self.fs = fs
        self.xs = list(range(xmin, xmax + 1))
        nfilt = len(self.xs)
        self.nfilt = nfilt
        self._fb = pyxSosFilterBank(nfilt, 5)
        for i, x in enumerate(self.xs):
            sos = self.designer.createSOSFilter(x)
            self._fb.setFilter(i, sos)

        self.xmin = xmin
        self.xmax = xmax
        self.N = 0

    def filter_(self, data):
        """
        Filter input data
        """
        assert data.ndim == 2
        assert data.shape[1] == 1, "invalid number of channels, should be 1"

        if data.shape[0] == 0:
            return {}

        filtered_data = self._fb.filter_(data)

        # Output given as a dictionary with nom_txt as the key
        output = {}

        tstart = self.N / self.fs
        Ncur = data.shape[0]
        tend = tstart + Ncur / self.fs

        t = np.linspace(tstart, tend, Ncur, endpoint=False)
        self.N += Ncur

        for i, x in enumerate(self.xs):
            # '31.5' to '16k'
            nom_txt = self.designer.nominal_txt(x)
            output[nom_txt] = {'t': t, 'data': filtered_data[:, [i]], 'x': x}

        return output

    def decimation(self, x):
        return [1]


class SosThirdOctaveFilterBank(SosFilterBank):
    """
    Filter bank which uses FIR filtering for each one-third octave frequency
    band.
    """

    def __init__(self, fs, xmin=None, xmax=None):
        """
        Initialize a second order sections filterbank.

        Args:
            fs: Sampling frequency [Hz]
            xmin: Minimum value for the bands
            xmax: Maximum value for the bands
        """
        self.designer = ThirdOctaveBankDesigner(fs)
        if xmin is None:
            xmin = self.designer.xs[0]
        if xmax is None:
            xmax = self.designer.xs[-1]
        SosFilterBank.__init__(self, fs, xmin, xmax)


class SosOctaveFilterBank(SosFilterBank):
    """
    Filter bank which uses FIR filtering for each one-third octave frequency
    band.
    """

    def __init__(self, fs, xmin=None, xmax=None):
        """
        Initialize a second order sections filterbank.

        Args:
            fs: Sampling frequency [Hz]
            xmin: Minimum value for the bands, if not specified, use minimum
            xmax: Maximum value for the bands, if not specified, use maximum
        """
        self.designer = OctaveBankDesigner(fs)
        if xmin is None:
            xmin = self.designer.xs[0]
        if xmax is None:
            xmax = self.designer.xs[-1]
        SosFilterBank.__init__(self, fs, xmin, xmax)


