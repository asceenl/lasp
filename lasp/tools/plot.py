#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description:
"""
__all__ = ['close', 'Figure', 'Bode', 'PS', 'PSD']

from .config import getReportQuality
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from lasp.lasp_common import (PLOT_COLORS_LIST, PLOT_NOCOLORS_LIST,
                              DEFAULT_FIGSIZE_H, DEFAULT_FIGSIZE_W)


class Figure:
    def __init__(self, **kwargs):
        ncols = kwargs.pop('ncols', 1)
        nrows = kwargs.pop('nrows', 1)
        color = kwargs.pop('color', (PLOT_NOCOLORS_LIST if getReportQuality()
                                     else PLOT_COLORS_LIST))
        if isinstance(color, bool):
            if color:
                color = PLOT_COLORS_LIST
            else:
                color = PLOT_NOCOLORS_LIST
        colors = cycler('color', color)

        figsize = kwargs.pop('figsize', (DEFAULT_FIGSIZE_W, DEFAULT_FIGSIZE_H))
        self._f = plt.figure(figsize=figsize)

        marker = kwargs.pop('marker', False)
        if marker:
            markers = cycler(marker=['o', 's', 'D', 'X', 'v', '^', '<', '>'])
        else:
            markers = cycler(marker=[None]*8)

        linewidths = cycler(linewidth=[1, 2, 1, 2, 2, 3, 2, 1])

        linestyles = cycler(
            linestyle=['-', '-', '--', ':', '-', '--', ':', '-.', ])

        self._ax = []
        self._legend = {}
        for row in range(nrows):
            self._legend[row] = {}
            for col in range(ncols):
                self._legend[row][col] = []
                ax = self._f.add_subplot(100*nrows
                                         + 10*ncols
                                         + (row*ncols + col)+1)
                ax.set_prop_cycle(
                    colors+linestyles+markers+linewidths)
                self._ax.append(ax)
        self._ncols = ncols
        self._cur_ax = self._ax[0]
        self._cur_col = 0
        self._cur_row = 0

        self._zorder = -1

    def setAx(self, row, col):
        self._cur_ax = self._ax[row*self._ncols+col]

    @property
    def fig(self):
        return self._f

    def markup(self):
        for ax in self._ax:
            ax.grid(True, 'both')
        self._zorder -= 1
        self.fig.show()

    def vline(self, x):
        self._ax[0].axvline(x)

    def plot(self, *args, **kwargs):
        line = self._cur_ax.plot(*args, **kwargs, zorder=self._zorder)
        self.markup()
        return line

    def loglog(self, *args, **kwargs):
        line = self._cur_ax.loglog(*args, **kwargs, zorder=self._zorder)
        self.markup()
        return line

    def semilogx(self, *args, **kwargs):
        line = self._cur_ax.semilogx(*args, **kwargs, zorder=self._zorder)
        self.markup()
        return line

    def xlabel(self, *args, **kwargs):
        all_ax = kwargs.pop('all_ax', False)
        if all_ax:
            for ax in self._ax:
                ax.set_xlabel(*args, **kwargs)
        else:
            self._cur_ax.set_xlabel(*args, **kwargs)

    def ylabel(self, *args, **kwargs):
        all_ax = kwargs.pop('all_ax', False)
        if all_ax:
            for ax in self._ax:
                ax.set_ylabel(*args, **kwargs)
        else:
            self._cur_ax.set_ylabel(*args, **kwargs)

    def legend(self, leg, *args, **kwargs):
        # all_ax = kwargs.pop('all_ax', False)
        if isinstance(leg, list) or isinstance(leg, tuple):
            self._legend[self._cur_col][self._cur_col] = list(leg)
        else:
            self._legend[self._cur_col][self._cur_col].append(leg)
        self._cur_ax.legend(self._legend[self._cur_col][self._cur_col])

    def savefig(self, *args, **kwargs):
        self.fig.savefig(*args, **kwargs)

    def xlim(self, *args, **kwargs):
        all_ax = kwargs.pop('all_ax', False)
        if all_ax:
            for ax in self._ax:
                ax.set_xlim(*args, **kwargs)
        else:
            self._cur_ax.set_xlim(*args, **kwargs)

    def ylim(self, *args, **kwargs):
        all_ax = kwargs.pop('all_ax', False)
        if all_ax:
            for ax in self._ax:
                ax.set_ylim(*args, **kwargs)
        else:
            self._cur_ax.set_ylim(*args, **kwargs)

    def title(self, *args, **kwargs):
        self._cur_ax.set_title(*args, **kwargs)

    def xticks(self, ticks):
        for ax in self._ax:
            ax.set_xticks(ticks)

    def close(self):
        plt.close(self._f)

    def xscale(self, scale):
        for ax in self._ax:
            ax.set_xscale(scale)


class Bode(Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(naxes=2, *args, **kwargs)

    def add(self, freq, phasor, qtyname='G', **kwargs):
        L = 20*np.log10(np.abs(phasor))
        phase = np.angle(phasor)*180/np.pi
        self.semilogx(freq, L, axno=0, **kwargs)
        self.semilogx(freq, phase, axno=1, **kwargs)
        self.ylabel('$L$ [%s] [dB]' % qtyname, axno=0)
        self.ylabel(fr'$\angle$ {qtyname} [$^\circ$]', axno=1)
        self.xlabel('Frequency [Hz]', axno=1)


class PS(Figure):
    def __init__(self, ref, *args, **kwargs):
        super().__init__(naxes=1, *args, **kwargs)
        self.ref = ref

    def add(self, fs, freq, ps, qtyname='C', **kwargs):

        overall = np.sum(ps)
        print(overall)
        overall_db = 10*np.log10(overall/self.ref**2)
        L = 10*np.log10(np.abs(ps)/self.ref**2)

        self.semilogx(freq, L, **kwargs)
        # self.plot(freq,L,**kwargs)
        self.ylabel('Level [dB re 20$\\mu$Pa]')
        self.xlabel('Frequency [Hz]')
        self.legend('%s. Overall SPL = %0.1f dB SPL' % (qtyname, overall_db))


class PSD(PS):
    def __init__(self, ref, *args, **kwargs):
        """
        Initialize a PSD plot

        Args:
            ref: Reference value for level in dB's

        """
        super().__init__(ref, *args, **kwargs)

    def add(self, fs, freq, ps, qtyname='C', **kwargs):
        df = freq[1]-freq[0]
        nfft = fs/df
        df = fs/nfft
        psd = ps / df

        overall = np.sum(np.abs(ps), axis=0)
        overall_db = 10*np.log10(overall/self.ref**2)
        L = 10*np.log10(abs(psd)/self.ref**2)

        self.semilogx(freq, L, **kwargs)
        self.ylabel('$L$ [%s] [dB re %0.0e]' % (qtyname, self.ref))
        self.xlabel('Frequency [Hz]')
        self.legend('%s. Overall SPL = %0.1f dB SPL' % (qtyname, overall_db))
