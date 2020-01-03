#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description: Filter design for frequency weighting curves (i.e. A and C 
weighting)
"""
from .fir_design import freqResponse, arbitrary_fir_design
import numpy as np

__all__ = ['A', 'C', 'A_fir_design', 'C_fir_design',
           'show_Afir', 'show_Cfir']

fr = 1000.
fL = 10**1.5
fH = 10**3.9
fLsq = fL**2
fHsq = fH**2
frsq = fr**2
fA = 10**2.45
D = np.sqrt(.5)

b = (1/(1-D))*(frsq+fLsq*fHsq/frsq-D*(fLsq+fHsq))
c = fLsq*fHsq
f2 = (3-np.sqrt(5.))/2*fA
f3 = (3+np.sqrt(5.))/2*fA
f1 = np.sqrt((-b-np.sqrt(b**2-4*c))/2)
f4 = np.sqrt((-b+np.sqrt(b**2-4*c))/2)
f4sq = f4**2


def A_uncor(f):
    """
    Computes the uncorrected frequency response of the A-filter
    """
    fsq = f**2
    num = f4sq*fsq**2
    denom1 = (fsq+f1**2)
    denom2 = np.sqrt((fsq+f2**2)*(fsq+f3**2))*(fsq+f4sq)

    return (num/(denom1*denom2))


def A(f):
    """
    Computes the linear A-weighting freqency response. Hence, to obtain
    A-weighted values, the *amplitude* need to be multiplied with this value.
    Hence, to correct dB levels, the value of 20*log(A) needs to be added to
    the level

    Args:
        f: Frequency array to compute values for
    Returns:
        A(f) for each frequency
    """
    Auncor = A_uncor(f)
    A1000 = A_uncor(1000.)
    return Auncor/A1000


def C_uncor(f):
    """
    Computes the uncorrected frequency response of the C-filter
    """
    fsq = f**2
    num = f4sq*fsq
    denom1 = (fsq+f1**2)
    denom2 = (fsq+f4**2)
    return num/(denom1*denom2)


def C(f):
    """
    Computes the linear A-weighting freqency response
    """
    Cuncor = C_uncor(f)
    C1000 = C_uncor(1000.)
    return Cuncor/C1000


def A_fir_design():
    fs = 48000.
    freq_design = np.linspace(0, 17e3, 3000)
    freq_design[-1] = fs/2
    amp_design = A(freq_design)
    amp_design[-1] = 0.

    L = 2048  # Filter order
    fir = arbitrary_fir_design(fs, L, freq_design, amp_design,
                               window='rectangular')
    return fir


def C_fir_design():
    fs = 48000.
    freq_design = np.linspace(0, 17e3, 3000)
    freq_design[-1] = fs/2
    amp_design = C(freq_design)
    amp_design[-1] = 0.

    L = 2048  # Filter order
    fir = arbitrary_fir_design(fs, L, freq_design, amp_design,
                               window='rectangular')
    return fir


def show_Afir():
    from asceefigs.plot import close, Figure
    close('all')

    fs = 48000.
    freq_design = np.linspace(0, 17e3, 3000)
    freq_design[-1] = fs/2
    amp_design = A(freq_design)
    amp_design[-1] = 0.
    firs = []

    # firs.append(arbitrary_fir_design(fs,L,freq_design,amp_design,window='hamming'))
    # firs.append(arbitrary_fir_design(fs,L,freq_design,amp_design,window='hann'))
    firs.append(A_fir_design())
    # from scipy.signal import iirdesign
    # b,a = iirdesign()
    freq_check = np.logspace(0, np.log10(fs/2), 5000)
    f = Figure()

    f.semilogx(freq_check, 20*np.log10(A(freq_check)))
    for fir in firs:
        H = freqResponse(fs, freq_check, fir)
        f.plot(freq_check, 20*np.log10(np.abs(H)))

    f.fig.get_axes()[0].set_ylim(-75, 3)


def show_Cfir():
    from asceefigs.plot import close, Figure
    close('all')

    fs = 48000.
    freq_design = np.linspace(0, 17e3, 3000)
    freq_design[-1] = fs/2
    amp_design = C(freq_design)
    amp_design[-1] = 0.
    firs = []

    # firs.append(arbitrary_fir_design(fs,L,freq_design,amp_design,window='hamming'))
    # firs.append(arbitrary_fir_design(fs,L,freq_design,amp_design,window='hann'))
    firs.append(C_fir_design())
    # from scipy.signal import iirdesign
    # b,a = iirdesign()
    freq_check = np.logspace(0, np.log10(fs/2), 5000)
    f = Figure()

    f.semilogx(freq_check, 20*np.log10(C(freq_check)))
    for fir in firs:
        H = freqResponse(fs, freq_check, fir)
        f.plot(freq_check, 20*np.log10(np.abs(H)))

    f.fig.get_axes()[0].set_ylim(-30, 1)
