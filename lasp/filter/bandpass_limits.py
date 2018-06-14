#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description: Limit lines for class 1 octave band filter limits according to
the ICS 17.140.50 standard.
"""
__all__ = ['G', 'fr', 'third_octave_band_limits', 'octave_band_limits']
import numpy as np

# Reference frequency
fr = 1000.
G = 10**(3/10)


def third_octave_band_limits(x):
    """
    Returns the class 1 third octave band filter limits for filter designator
    x.

    Args:
        x: Filter offset power from the reference frequency of 1000 Hz.

    Returns:
        freq, ulim, llim: Tuple of Numpy arrays containing the frequencyies,
        upper and lower limits of the filter.
    """
    b = 3

    fm = G**(x/b)*fr
    plusinf = 20
    f_ratio_pos = [1., 1.02667, 1.05575, 1.08746, 1.12202, 1.12202,
                   1.29437, 1.88173, 3.05365, 5.39195, plusinf]

    f_ratio_neg = [0.97402, 0.94719, 0.91958, 0.89125, 0.89125,
                   0.77257, 0.53143, 0.32748, 0.18546, 1/plusinf]
    f_ratio_neg.reverse()

    f_ratio = f_ratio_neg + f_ratio_pos

    mininf = -1e300

    upper_limits_pos = [.3]*5 + [-2, -17.5, -42, -61, -70, -70]
    upper_limits_neg = upper_limits_pos[:]
    upper_limits_neg.reverse()
    upper_limits = np.array(upper_limits_neg[:-1] + upper_limits_pos)

    lower_limits_pos = [-.3, -.4, -.6, -1.3, -5, -5, mininf, mininf,
                        mininf, mininf, mininf]
    lower_limits_neg = lower_limits_pos[:]
    lower_limits_neg.reverse()
    lower_limits = np.array(lower_limits_neg[:-1] + lower_limits_pos)

    freqs = fm*np.array(f_ratio)

    return freqs, upper_limits, lower_limits


def octave_band_limits(x):

    b = 1

    # Exact midband frequency
    fm = G**(x/b)*fr

    G_power_values_pos = [0, 1/8, 1/4, 3/8, 1/2, 1/2, 1, 2, 3, 4]
    G_power_values_neg = [-i for i in G_power_values_pos]
    G_power_values_neg.reverse()
    G_power_values = G_power_values_neg[:-1] + G_power_values_pos

    mininf = -1e300

    lower_limits_pos = [-0.3, -0.4, -0.6, -1.3, -5.0, -5.0] + 4*[mininf]
    lower_limits_neg = lower_limits_pos[:]
    lower_limits_neg.reverse()
    lower_limits = np.asarray(lower_limits_neg[:-1] + lower_limits_pos)

    upper_limits_pos = [0.3]*5 + [-2, -17.5, -42, -61, -70]
    upper_limits_neg = upper_limits_pos[:]
    upper_limits_neg.reverse()
    upper_limits = np.asarray(upper_limits_neg[:-1] + upper_limits_pos)

    freqs = fm*G**np.asarray(G_power_values)

    return freqs, upper_limits, lower_limits


if __name__ == '__main__':

    from asceefigs.plot import close, Figure
    close('all')
    freqs, upper_limits, lower_limits = octave_band_limits(0)

    f = Figure()
    f.semilogx(freqs, lower_limits)
    f.semilogx(freqs, upper_limits)

    f.ylim(-80, 1)
