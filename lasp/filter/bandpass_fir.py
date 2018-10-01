#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description: FIR filter design for octave bands from 16Hz to 16 kHz for a
sampling frequency of 48 kHz, FIR filter design for one-third octave bands

See test/octave_fir_test.py for a testing

"""
from .fir_design import bandpass_fir_design, freqResponse as frsp
import numpy as np
__all__ = ['OctaveBankDesigner', 'ThirdOctaveBankDesigner']


class FilterBankDesigner:
    """
    A class responsible for designing FIR filters
    """
    G = 10**(3/10)
    fr = 1000.
    L = 256  # Filter order
    fs = 48000.  # Sampling frequency

    def fm(self, x):
        """
        Returns the exact midband frequency of the bandpass filter

        Args:
            x: Midband designator
        """
        # Exact midband frequency
        return self.G**(x/self.b)*self.fr

    def fl(self, x):
        """
        Returns the exact cut-on frequency of the bandpass filter

        Args:
            x: Midband designator
        """
        return self.fm(x)*self.G**(-1/(2*self.b))

    def fu(self, x):
        """
        Returns the exact cut-off frequency of the bandpass filter

        Args:
            x: Midband designator
        """
        return self.fm(x)*self.G**(1/(2*self.b))

    def createFilter(self, fs, x):
        """
        Create a FIR filter for band designator b and sampling frequency fs.
        Decimation should be obtained from decimation() method.

        Returns:
            filter: 1D ndarray with FIR filter coefficients
        """
        assert np.isclose(fs, self.fs), "Invalid sampling frequency"
        fd = fs / np.prod(self.decimation(x))

        # For designing the filter, the lower and upper frequencies need to be
        # slightly adjusted to fall within the limits for a class 1 filter.
        fl = self.fl(x)*self.fac_l(x)
        fu = self.fu(x)*self.fac_u(x)

        return bandpass_fir_design(self.L, fd, fl, fu)

    def freqResponse(self, fs, x, freq):
        """
        Compute the frequency response for a certain filter

        Args:
            fs: Sampling frequency [Hz]
            x: Midband designator
        """
        fir = self.createFilter(fs, x)

        # Decimated sampling frequency [Hz]
        fd = fs / np.prod(self.decimation(x))

        return frsp(fd, freq, fir)

    def nominal_txt_tox(self, nom_txt):
        """
        Returns the x-value corresponding to a certain nominal txt: '1k' -> 0
        """
        for x in self.xs:
            if self.nominal_txt(x) == nom_txt:
                return x
        raise ValueError(f'Could not find an x-value corresponding to {nom_txt}.')

class OctaveBankDesigner(FilterBankDesigner):
    """
    Octave band filter designer
    """
    def __init__(self):
        pass

    @property
    def b(self):
        # Band division, 1 for octave bands
        return 1

    @property
    def xs(self):
        return list(range(-6, 5))

    def nominal_txt(self, x):
        # Text corresponding to the nominal frequency
        nominals = {4: '16k',
                    3: '8k',
                    2: '4k',
                    1: '2k',
                    0: '1k',
                    -1: '500',
                    -2: '250',
                    -3: '125',
                    -4: '63',
                    -5: '31.5',
                    -6: '16'}
        return nominals[x]

    def fac_l(self, x):
        """
        Factor with which to multiply the cut-on frequency of the FIR filter
        """
        if x == 4:
            return .995
        elif x in (3, 1):
            return .99
        elif x in(-6, -4, -2, 2, 0):
            return .98
        else:
            return .96

    def fac_u(self, x):
        """
        Factor with which to multiply the cut-off frequency of the FIR filter
        """
        if x == 4:
            return 1.004
        elif x in (3, 1):
            return 1.006
        elif x in (-6, -4, -2, 2, 0):
            return 1.01
        else:
            return 1.02

    def decimation(self, x):
        """
        Required decimation for each filter
        """
        if x > 1:
            return [1]
        elif x > -2:
            return [4]
        elif x > -4:
            return [4, 4]
        elif x > -6:
            return [4, 4, 4]
        elif x == -6:
            return [4, 4, 4, 4]
        assert False, 'Overlooked decimation'


class ThirdOctaveBankDesigner(FilterBankDesigner):

    def __init__(self):
        self.xs = list(range(-16, 14))
        # Text corresponding to the nominal frequency
        self._nominal_txt = ['25', '31.5', '40',
                            '50', '63', '80',
                            '100', '125', '160',
                            '200', '250', '315',
                            '400', '500', '630',
                            '800', '1k', '1.25k',
                            '1.6k', '2k', '2.5k',
                            '3.15k', '4k', '5k',
                            '6.3k', '8k', '10k',
                            '12.5k', '16k', '20k']

        assert len(self.xs) == len(self._nominal_txt)

    @property
    def b(self):
        # Band division factor, 3 for one-third octave bands
        return 3

    def nominal_txt(self, x):
        # Put the nominal frequencies in a dictionary for easy access with
        # x as the key.
        index = x - self.xs[0]
        return self._nominal_txt[index]

    @staticmethod
    def decimation(x):
        if x > 5:
            return [1]
        elif x > -1:
            return [4]
        elif x > -7:
            return [4, 4]
        elif x > -13:
            return [4, 4, 4]
        elif x > -17:
            return [4, 4, 4, 4]
        assert False, 'Bug: overlooked decimation'

    @staticmethod
    def fac_l(x):
        if x in (-13, -7, -1, 5, 11, 12, 13):
            return .995
        elif x in (-12, -6, 0, 6):
            return .98
        else:
            return .99

    @staticmethod
    def fac_u(x):
        if x in (-14, -13, -8, -7, -1, -2, 3, 4, 5, 10, 11, 12):
            return 1.005
        elif x in (12, 13):
            return 1.003
        elif x in (12, -6, 0):
            return 1.015
        else:
            return 1.01
