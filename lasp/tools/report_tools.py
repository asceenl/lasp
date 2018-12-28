#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: J.A. de Jong - ASCEE

Description: backend tools for easy postprocessing of measurements
"""
from .plot import Figure
from lasp.wrappers import AvPowerSpectra
from lasp.lasp_measurement import Measurement
from lasp.lasp_common import (FreqWeighting, TimeWeighting,
                              getFreq, getTime, Window, P_REF)
from lasp.lasp_weighcal import WeighCal
from lasp.lasp_octavefilter import OctaveFilterBank, ThirdOctaveFilterBank
from lasp.lasp_figuredialog import FigureDialog
from lasp.lasp_figure import Plotable, PlotOptions
from lasp.lasp_slm import SLM
import numpy as np
import sys


def close():
    import matplotlib.pyplot as plt
    plt.close('all')


def PSPlot(fn_list, **kwargs):
    """
    Create a power spectral density plot, ASCEE style

    Args:
        fn_list: list of measurement filenames to plot PSD for
        fw:
        fs:
        nfft:
        xscale:
        yscale:
    """

    fw = kwargs.pop('fw', FreqWeighting.A)
    nfft = kwargs.pop('nfft', 2048)
    xscale = kwargs.pop('xscale', 'log')
    yscale = kwargs.pop('yscale', 'PSD')
    ylim = kwargs.pop('ylim', (0, 100))
    xlim = kwargs.pop('xlim', (100, 10000))
    f = Figure(**kwargs)

    print(kwargs)
    if xscale == 'log':
        pltfun = f.semilogx
    else:
        pltfun = f.plot

    for fn in fn_list:
        meas = Measurement(fn)
        fs = meas.samplerate
        data = meas.praw()
        aps = AvPowerSpectra(nfft, 1, 50.)
        weighcal = WeighCal(fw, nchannels=1,
                            fs=fs)
        weighted = weighcal.filter_(data)
        ps = aps.addTimeData(weighted)
        freq = getFreq(fs, nfft)
        if yscale == 'PSD':
            df = fs/nfft
            type_str = '/$\\sqrt{\\mathrm{Hz}}$'
        elif yscale == 'PS':
            df = 1.
            type_str = ''
        else:
            raise ValueError("'type' should be either 'PS' or 'PSD'")

        psd_log = 10*np.log10(ps[:, 0, 0].real/df/2e-5**2)

        pltfun(freq, psd_log)

    f.xlabel('Frequency [Hz]')
    f.ylabel(f'Level [dB({fw[0]}) re (20$\\mu$ Pa){type_str}')
    f.ylim(ylim)
    f.xlim(xlim)
    return f


def PowerSpectra(fn_list, **kwargs):
    nfft = kwargs.pop('nfft', 2048)
    window = kwargs.pop('window', Window.hann)
    fw = kwargs.pop('fw', FreqWeighting.A)
    overlap = kwargs.pop('overlap', 50.)
    ptas = []
    for fn in fn_list:
        meas = Measurement(fn)
        fs = meas.samplerate
        weighcal = WeighCal(fw, nchannels=1,
                            fs=fs, calfile=None)
        praw = meas.praw()
        weighted = weighcal.filter_(praw)
        aps = AvPowerSpectra(nfft, 1, overlap, window[0])
        result = aps.addTimeData(weighted)[:, 0, 0].real
        pwr = 10*np.log10(result/P_REF**2)

        freq = getFreq(fs, nfft)
        ptas.append(Plotable(freq, pwr))

        pto = PlotOptions.forPower()
        pto.ylabel = f'L{fw[0]} [dB({fw[0]})]'
    return ptas


def Levels(fn_list, **kwargs):

    bank = kwargs.pop('bank', 'third')
    fw = kwargs.pop('fw', FreqWeighting.A)
    tw = kwargs.pop('tw', TimeWeighting.fast)
    xmin_txt = kwargs.pop('xmin', '100')
    xmax_txt = kwargs.pop('xmax', '16k')

    levels = []
    leveltype = 'eq'

    for fn in fn_list:
        meas = Measurement(fn)
        fs = meas.samplerate
        weighcal = WeighCal(fw, nchannels=1,
                            fs=fs, calfile=None)
        praw = meas.praw()
        weighted = weighcal.filter_(praw)
        if bank == 'third':
            filt = ThirdOctaveFilterBank(fs)
            xmin = filt.nominal_txt_tox(xmin_txt)
            xmax = filt.nominal_txt_tox(xmax_txt)

        elif bank == 'overall':
            slm = SLM(meas.samplerate, tw)
            slm.addData(weighted)
            levels.append(
                Plotable(' ', slm.Lmax if leveltype == 'max' else slm.Leq,
                         name=meas.name))
            continue
        else:
            raise NotImplementedError()

        # Octave bands
        # filt = OctaveFilterBank(fs)
        filtered_out = filt.filter_(weighted)
        level = np.empty((xmax - xmin + 1))
        xlabels = []
        for i, x in enumerate(range(xmin, xmax+1)):
            nom = filt.nominal_txt(x)
            xlabels.append(nom)
            filt_x = filtered_out[nom]['data']
            slm = SLM(filt.fs, tw)
            slm.addData(filt_x)
            leveli = slm.Lmax if leveltype == 'max' else slm.Leq
            level[i] = leveli
        levels.append(Plotable(xlabels, level, name=meas.name))
    return levels


def LevelDifference(levels):
    assert len(levels) == 2
    return Plotable(name='Difference', x=levels[0].x,
                    y=levels[1].y-levels[0].y)


def LevelFigure(levels, show=True, **kwargs):
    figtype = kwargs.pop('figtype', 'bar')
    from PySide.QtGui import QApplication, QFont
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    app.setFont(QFont('Linux Libertine'))
    size = kwargs.pop('size', (1200, 600))
    if figtype == 'bar':
        opts = PlotOptions.forLevelBars()
    elif figtype == 'line':
        opts = PlotOptions.forPower()
    else:
        raise RuntimeError('figtype should be either line or bar')
    opts.ylim = kwargs.pop('ylim', (0, 100))
    opts.ylabel = kwargs.pop('ylabel', 'LAeq [dB(A)]')
    opts.xlabel = kwargs.pop('xlabel', None)
    opts.legend = kwargs.pop('legend', [level.name for level in levels])
    opts.legendpos = kwargs.pop('legendpos', None)
    opts.title = kwargs.pop('title', None)

    def Plotter(ptas, pto):
        fig = FigureDialog(None, None, pto, figtype)
        for pta in ptas:
            fig.fig.add(pta)
        return fig

    fig = Plotter(levels, opts)
    if show:
        fig.show()
    fig.resize(*size)
    if show:
        app.exec_()
    return fig
