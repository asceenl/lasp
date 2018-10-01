#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description:
"""
import matplotlib as mpl
mpl.use('Qt5Agg')
preamble = [
      r'\usepackage{libertine-type1}'
      r'\usepackage[libertine]{newtxmath}'
#          r'\usepackage{fontspec}',
      #r'\setmainfont{Libertine}',
]
params = {
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.unicode': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': preamble,
}
mpl.rcParams.update(params)

from asceefigs.config import report_quality
# report_quality()

from lasp.lasp_measurement import Measurement
from lasp.lasp_weighcal import WeighCal, FreqWeighting
from lasp.lasp_slm import SLM, TimeWeighting, P_REF
from lasp.wrappers import AvPowerSpectra
from lasp.lasp_common import getFreq
import os
import sys
import numpy as np
from asceefigs.plot import Figure, PS, close
import matplotlib.pyplot as pl
import time

#close('all')


def select_file():
    measfiles = []
    for path, name, filenames in os.walk(os.getcwd()):
        for filename in filenames:
            if '.h5' in filename:
                measfiles.append(os.path.join(path, filename))

    if len(measfiles) == 0:
        raise ValueError('No measurement files in current directory')

    for i, mfile in enumerate(measfiles):
        print('- %20s: %i' % (mfile, i))
    no = input('Please select a file (%i - %i):' % (0, len(measfiles)))
    1
    no = int(no)
    return measfiles[no]


fn = sys.argv[2]
# type_ : spec, sp
type_ = sys.argv[1]

#else:
#    fn = select_file()

##
#fn = 'Cees_1'
#fn = 'holland_1'
fs = 48000

meas = Measurement(fn)

ts = meas.time
print('Measurement time: ', time.ctime(ts))

praw = meas.praw()[:, [0]]
N = praw.shape[0]


if type_ == 'spec':
    weighcal_slmA = WeighCal(FreqWeighting.A)
    weighcal_slmC = WeighCal(FreqWeighting.C)

    filtered_A = weighcal_slmA.filter_(praw)
    filtered_C = weighcal_slmC.filter_(praw)
    # %% Sound level meter
    tw = TimeWeighting.fast

    slm_A = SLM(fs, tw=tw)
    slm_C = SLM(fs, tw=tw)

    LAF = slm_A.addData(filtered_A)
    LCF = slm_C.addData(filtered_C)

    # Strip off filter initialization part
    LAF = LAF[int(tw[0]*fs):, :]
    LCF = LCF[int(tw[0]*fs):, :]
    N = LAF.shape[0]

    t = np.linspace(0, N/fs, N, False)
    Lfig = pl.figure(figsize=(8, 8))

#    from matplotlib import gridspec
    #gs = gridspec.GridSpec(2, 2, height_ratios=(3, 1), width_ratios=(9, 1))
    #Lax = pl.subplot(gs[0,0])
    ax = Lfig.add_subplot(211)
    pl.title('')
    pl.plot(t, LAF[:, 0])
    pl.plot(t, LCF[:, 0])
    pl.ylabel('Level [dB]')
    pl.legend(['LAF', 'LCF'])
    pl.ylim(40, 80)
    pl.xlim(0, meas.T)
    pl.grid('on', 'both')

    print('Maximum A-level:', slm_A.Lmax)
    print('Maximum C-level:', slm_C.Lmax)

    from scipy.signal import spectrogram
    nfft = 8192
    noverlap = 4*nfft//5
    freq_sp, t_sp, Sxx = spectrogram(filtered_C[:, 0], fs, scaling='spectrum',
                                     window=(
        'hann'), nperseg=nfft, noverlap=noverlap)
    # Chop off higher frequencies
    
    nfreq = freq_sp.shape[0]
    nf_start = 5
    nf_end = int(nfreq/40)
    freq_sp = freq_sp[nf_start:nf_end]
    Sxx = Sxx[nf_start:nf_end, :]
    Sxxpow = 10*np.log10(Sxx/P_REF**2)
    maxSxx = np.max(Sxxpow)
    minSxx = np.min(Sxxpow)
    
    print(f'max Sxx: {maxSxx:.1f} dB')
    print(f'min Sxx: {minSxx:.1f} dB')
 
    #SPax = pl.subplot(gs[1,0])

    ax1 = Lfig.add_subplot(212)
    pl.title('C-weighted spectrogram')
    colormesh = pl.pcolormesh(t_sp, freq_sp, Sxxpow,
                              cmap='rainbow', vmin=30, vmax=60)
    pl.xlabel('Time [s]')
    pl.ylabel('Frequency [Hz]')
    pl.xlim(0, meas.T)
    pl.yscale('linear')

    # %% Colorbar
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(ax1,
                       width="5%",  # width = 10% of parent_bbox width
                       height="100%",  # height : 50%
                       loc=6,
                       bbox_to_anchor=(1.02, 0., 1, 1),
                       bbox_transform=ax1.transAxes,
                       borderpad=0,
                       )
    #pl.colorbar(orientation='horizontal')
    #cax = pl.subplot(gs[1,1])
    cb = pl.colorbar(colormesh, cax=axins)
    fn_base = os.path.splitext(fn)[0]
#    pl.savefig('%s.eps' %fn,dpi=600)
    pl.show(block=False)

#pl.subplot2grid((2,2),(1,1))

#pl.tight_layout()

#l1 = pl.axvline(2,color='k')
#pl.axvline(3,color='k')
#del l1
# %%
else:
    # %%
    weighcal_A = WeighCal(FreqWeighting.A,
                          calfile='/home/anne/wip/UMIK-1/cal/7027430_90deg.txt',
                          sens=0.053690387255872614)

    filtered_A = weighcal_A.filter_(praw)
    nfft = 8192
    aps = AvPowerSpectra(nfft, 1, 50.)
    ps = aps.addTimeData(filtered_A)
    psplot = PS(P_REF)
    freq = getFreq(fs, nfft)
    psplot.add(fs, freq, ps[:, 0, 0], 'A-gewogen')
    psplot.xlim(10, 10000)
    psplot.ylim(20, 80)
    psplot.savefig('%s_sp.eps' % fn)
    pl.show()
    # %%

input('Press any key to close')
pl.savefig(f'{fn_base}_spectrogram.png', dpi=600)
pl.savefig(f'{fn_base}_spectrogram.eps')