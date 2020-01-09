#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description: FIR filter design for octave bands from 16Hz to 16 kHz for a
sampling frequency of 48 kHz, filter design for one-third octave bands.
Resulting filters are supposed to be standard compliant.

See test/octave_fir_test.py for a testing

"""
from .fir_design import bandpass_fir_design, freqResponse as firFreqResponse
import numpy as np

# For designing second-order sections
from scipy.signal import butter


__all__ = ['OctaveBankDesigner', 'ThirdOctaveBankDesigner']


class FilterBankDesigner:
    """A class responsible for designing FIR filters."""

    def __init__(self, fs):
        """Initialize a filter bank designer.

        Args:
            fs: Sampling frequency [Hz]
        """
        # Default FIR filter length
        firFilterLength = 256  # Filter order
        self.fs = fs

        # Constant G, according to standard
        self.G = 10**(3/10)

        # Reference frequency for all filter banks
        self.fr = 1000.

    def testStandardCompliance(self, x, freq, h_dB, filter_class=0):
        """Test whether the filter with given frequency response is compliant 
        with the standard.

        Args:
            x: Band designator
            freq: Array of frequencies to test for. Note: *Should be fine
            enough to follow response!*
            h_dB: Filter frequency response in *deciBell*
            filter_class: Filter class to test for

        Returns:
            True if filter is norm-compliant, False if not
        
        """
        # Skip zero-frequency
        if np.isclose(freq[0], 0):
            freq = freq[1:]
            h_dB = h_dB[1:]
        freqlim, llim, ulim = self.band_limits(x, filter_class)

        # Interpolate limites to frequency array as given
        llim_full = np.interp(freq, freqlim, llim, left=-np.inf, right=-np.inf)
        ulim_full = np.interp(freq, freqlim, ulim, left=ulim[0], right=ulim[-1])

        return bool(np.all(llim_full <= h_dB) and
                    np.all(ulim_full >= h_dB))

    def fm(self, x):
        """Returns the exact midband frequency of the bandpass filter.

        Args:
            x: Midband designator
        """
        # Exact midband frequency
        return self.G**(x/self.b)*self.fr

    def fl(self, x):
        """Returns the exact cut-on frequency of the bandpass filter.

        Args:
            x: Midband designator
        """
        return self.fm(x)*self.G**(-1/(2*self.b))

    def fu(self, x):
        """Returns the exact cut-off frequency of the bandpass filter.

        Args:
            x: Midband designator
        """
        return self.fm(x)*self.G**(1/(2*self.b))

    def createFirFilter(self, x):
        """Create a FIR filter for band designator b and sampling frequency fs.
        firdecimation should be obtained from firdecimation() method.

        Returns:
            filter: 1D ndarray with FIR filter coefficients
        """
        assert np.isclose(fs, self.fs), "Invalid sampling frequency"
        fd = fs / np.prod(self.firDecimation(x))

        # For designing the filter, the lower and upper frequencies need to be
        # slightly adjusted to fall within the limits for a class 1 filter.
        fl = self.fl(x)*self.firFac_l(x)
        fu = self.fu(x)*self.firFac_u(x)

        return bandpass_fir_design(self.firFilterLength, fd, fl, fu)

    def createSOSFilter(self, x: int):
        """Create a Second Order Section filter (cascaded BiQuad's) for the
        given sample rate and band designator.

        Args:
            x: Band designator
        """

        SOS_ORDER = 5

        fs = self.fs
        fl = self.fl(x)*self.sosFac_l(x)
        fu = self.fu(x)*self.sosFac_u(x)

        fnyq = fs/2

        # Normalized upper and lower frequencies of the bandpass
        fl_n = fl/fnyq
        fu_n = fu/fnyq

        return butter(SOS_ORDER, [fl_n, fu_n], output='sos', btype='band')

    def firFreqResponse(self, x, freq):
        """Compute the frequency response for a certain filter.

        Args:
            x: Midband designator
            freq: Array of frequencies to evaluate on

        Returns:
            h: Linear filter transfer function [-]
        """
        fir = self.createFirFilter(fs, x)

        # Decimated sampling frequency [Hz]
        fd = fs / np.prod(self.firdecimation(x))

        return firFreqResponse(fd, freq, fir)

    def nominal_txt_tox(self, nom_txt: str):
        """Returns the x-value corresponding to a certain nominal txt: '1k' ->
        0.

        Args:
            nom_txt: Text-representation of midband frequency
        """
        for x in self.xs:
            if self.nominal_txt(x) == nom_txt:
                return x
        raise ValueError(
            f'Could not find an x-value corresponding to {nom_txt}.')


class OctaveBankDesigner(FilterBankDesigner):
    """Octave band filter designer."""

    def __init__(self, fs):
        super().__init__(fs)

    @property
    def b(self):
        # Band division, 1 for octave bands
        return 1

    @property
    def xs(self):
        """All possible band designators for an octave band filter."""
        return list(range(-6, 5))

    def band_limits(self, x, filter_class=0):
        """Returns the octave band filter limits for filter designator x.

        Args:
            x: Filter offset power from the reference frequency of 1000 Hz.
            filter_class: Either 0 or 1, defines the tolerances on the frequency
            response

        Returns:
            freq, llim, ulim: Tuple of Numpy arrays containing the frequencies of
            the corner points of the filter frequency response limits, lower limits
            in *deciBell*, upper limits in *deciBell*, respectively.
        """
        b = 1

        # Exact midband frequency
        fm = self.G**(x/self.b)*self.fr

        G_power_values_pos = [0, 1/8, 1/4, 3/8, 1/2, 1/2, 1, 2, 3, 4]
        G_power_values_neg = [-i for i in G_power_values_pos]
        G_power_values_neg.reverse()
        G_power_values = G_power_values_neg[:-1] + G_power_values_pos

        mininf = -1e300

        if filter_class == 1:
            lower_limits_pos = [-0.3, -0.4, -0.6, -1.3, -5.0, -5.0] + 4*[mininf]
        elif filter_class == 0:
            lower_limits_pos = [-0.15, -0.2, -0.4, -1.1, -4.5, -4.5] + 4*[mininf]
        lower_limits_neg = lower_limits_pos[:]
        lower_limits_neg.reverse()
        lower_limits = np.asarray(lower_limits_neg[:-1] + lower_limits_pos)

        if filter_class == 1:
            upper_limits_pos = [0.3]*5 + [-2, -17.5, -42, -61, -70]
        if filter_class == 0:
            upper_limits_pos = [0.15]*5 + [-2.3, -18, -42.5, -62, -75]
        upper_limits_neg = upper_limits_pos[:]
        upper_limits_neg.reverse()
        upper_limits = np.asarray(upper_limits_neg[:-1] + upper_limits_pos)

        freqs = fm*self.G**np.asarray(G_power_values)

        return freqs, lower_limits, upper_limits


    def nominal_txt(self, x):
        """Returns textual repressentation of corresponding to the nominal
        frequency."""
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
        assert len(nominals) == len(self.xs)
        return nominals[x]

    def firFac_l(self, x):
        """Factor with which to multiply the cut-on frequency of the FIR
        filter."""
        assert int(self.fs) == 48000, 'Fir coefs are only valid for 48kHz fs'
        if x == 4:
            return .995
        elif x in (3, 1):
            return .99
        elif x in(-6, -4, -2, 2, 0):
            return .98
        else:
            return .96

    def firFac_u(self, x):
        """Factor with which to multiply the cut-off frequency of the FIR
        filter."""
        assert int(self.fs) == 48000, 'Fir coefs are only valid for 48kHz fs'
        if x == 4:
            return 1.004
        elif x in (3, 1):
            return 1.006
        elif x in (-6, -4, -2, 2, 0):
            return 1.01
        else:
            return 1.02

    def firDecimation(self, x):
        """Required firdecimation for each filter."""
        assert int(self.fs) == 48000, 'Fir coefs are only valid for 48kHz fs'
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
        assert False, 'Overlooked firdecimation'

    def sosFac_l(self, x):
        """Left side percentage of change in cut-on frequency for designing the
        filter, for OCTAVE band filter.

        Args:
            x: Filter band designator
        """
        # Idea: correct for frequency warping:
        if int(self.fs) in [48000, 96000]:
            return 1.0
        else:
            raise ValueError('Unimplemented sampling frequency for SOS'
                             'filter design')

    def sosFac_u(self, x):
        """Right side percentage of change in cut-on frequency for designing
        the filter.

        Args:
            x: Filter band designator
        """
        if int(self.fs) in [48000, 96000]:
            return 1.0
        else:
            raise ValueError('Unimplemented sampling frequency for SOS'
                             'filter design')


class ThirdOctaveBankDesigner(FilterBankDesigner):

    def __init__(self, fs):
        super().__init__(fs)
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

    def band_limits(self, x, filter_class=0):
        """Returns the third octave band filter limits for filter designator x.

        Args:
            x: Filter offset power from the reference frequency of 1000 Hz.
            filter_class: Either 0 or 1, defines the tolerances on the frequency
            response

        Returns:
            freq, llim, ulim: Tuple of Numpy arrays containing the frequencies of
            the corner points of the filter frequency response limits, lower limits
            in *deciBell*, upper limits in *deciBell*, respectively.
        """

        fm = self.G**(x/self.b)*self.fr
        plusinf = 20
        f_ratio_pos = [1., 1.02667, 1.05575, 1.08746, 1.12202, 1.12202,
                       1.29437, 1.88173, 3.05365, 5.39195, plusinf]

        f_ratio_neg = [0.97402, 0.94719, 0.91958, 0.89125, 0.89125,
                       0.77257, 0.53143, 0.32748, 0.18546, 1/plusinf]
        f_ratio_neg.reverse()

        f_ratio = f_ratio_neg + f_ratio_pos

        mininf = -1e300

        if filter_class == 1:
            upper_limits_pos = [.3]*5 + [-2, -17.5, -42, -61, -70, -70]
        elif filter_class == 0:
            upper_limits_pos = [.15]*5 + [-2.3, -18, -42.5, -62, -75, -75]
        else:
            raise ValueError('Filter class should either be 0 or 1')

        upper_limits_neg = upper_limits_pos[:]
        upper_limits_neg.reverse()
        upper_limits = np.array(upper_limits_neg[:-1] + upper_limits_pos)

        if filter_class == 1:
            lower_limits_pos = [-.3, -.4, -.6, -1.3, -5, -5, mininf, mininf,
                                mininf, mininf, mininf]
        elif filter_class == 0:
            lower_limits_pos = [-.15, -.2, -.4, -1.1, -4.5, -4.5, mininf, mininf,
                                mininf, mininf, mininf]

        lower_limits_neg = lower_limits_pos[:]
        lower_limits_neg.reverse()
        lower_limits = np.array(lower_limits_neg[:-1] + lower_limits_pos)

        freqs = fm*np.array(f_ratio)

        return freqs, lower_limits, upper_limits

    def firDecimation(self, x):
        assert int(self.fs) == 48000, 'Fir coefs are only valid for 48kHz fs'
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
        assert False, 'Bug: overlooked firdecimation'

    def firFac_l(self, x):
        if x in (-13, -7, -1, 5, 11, 12, 13):
            return .995
        elif x in (-12, -6, 0, 6):
            return .98
        else:
            return .99

    def firFac_u(self, x):
        if x in (-14, -13, -8, -7, -1, -2, 3, 4, 5, 10, 11, 12):
            return 1.005
        elif x in (12, 13):
            return 1.003
        elif x in (12, -6, 0):
            return 1.015
        else:
            return 1.01

    def sosFac_l(self, x):
        """Left side percentage of change in cut-on frequency for designing the
        filter."""
        # Idea: correct for frequency warping:
        if np.isclose(self.fs, 48000):
            return 1.00
        else:
            raise ValueError('Unimplemented sampling frequency for SOS'
                             'filter design')

    def sosFac_u(self, x):
        """Right side percentage of change in cut-on frequency for designing
        the filter."""
        if np.isclose(self.fs, 48000):
            return 1
        else:
            raise ValueError('Unimplemented sampling frequency for SOS'
                             'filter design')
