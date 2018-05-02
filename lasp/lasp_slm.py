#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound level meter implementation
@author: J.A. de Jong - ASCEE
"""
from .wrappers import SPLowpass
from .lasp_computewidget import ComputeWidget
import numpy as np
from .lasp_config import zeros
from .lasp_common import (FreqWeighting, sens, calfile,
                          TimeWeighting, getTime, P_REF)
from .lasp_weighcal import WeighCal
from .lasp_gui_tools import wait_cursor
from .lasp_figure import FigureDialog, PlotOptions
from .ui_slmwidget import Ui_SlmWidget

__all__ = ['SLM', 'SlmWidget']


class Dummy:
    """
    Emulate filtering, but does not filter anything at all.
    """

    def filter_(self, data):
        return data[:, np.newaxis]


class SLM:
    """
    Sound Level Meter, implements the single pole lowpass filter
    """

    def __init__(self, fs, weighcal,
                 tw=TimeWeighting.default,
                 ):
        """
        Initialize a sound level meter object. Number of channels comes from
        the weighcal object

        Args:
            fs: Sampling frequency [Hz]
            weighcal: WeighCal instance used for calibration and frequency
            weighting.
            nchannels: Number of channels to allocate filters for
        """
        nchannels = weighcal.nchannels
        self.nchannels = nchannels
        self._weighcal = weighcal
        if tw[0] is not TimeWeighting.none[0]:
            self._lps = [SPLowpass(fs, tw[0]) for i in range(nchannels)]
        else:
            self._lps = [Dummy() for i in range(nchannels)]
        self._Lmax = zeros(nchannels)

    @property
    def Lmax(self):
        """
        Returns the currently maximum recorded level
        """
        return self._Lmax

    def addData(self, data):
        """
        Add new fresh timedata to the Sound Level Meter

        Args:
            data:
        """
        if data.ndim == 1:
            data = data[:, np.newaxis]

        data_weighted = self._weighcal.filter_(data)

        # Squared
        sq = data_weighted**2
        if sq.shape[0] == 0:
            return np.array([])

        tw = []

        # Time-weight the signal
        for chan, lp in enumerate(self._lps):
            tw.append(lp.filter_(sq[:, chan])[:, 0])

        tw = np.asarray(tw).transpose()

        Level = 10*np.log10(tw/P_REF**2)

        curmax = np.max(Level)
        if curmax > self._Lmax:
            self._Lmax = curmax

        return Level


class SlmWidget(ComputeWidget, Ui_SlmWidget):
    def __init__(self):
        """
        Initialize the SlmWidget.
        """
        super().__init__()
        self.setupUi(self)
        FreqWeighting.fillComboBox(self.freqweighting)

        self.setMeas(None)

    def setMeas(self, meas):
        """
        Set the current measurement for this widget.

        Args:
            meas: if None, the Widget is disabled
        """
        self.meas = meas
        if meas is None:
            self.setEnabled(False)
        else:
            self.setEnabled(True)
            rt = meas.recTime
            self.starttime.setRange(0, rt, 0)
            self.stoptime.setRange(0, rt, rt)

            self.channel.clear()
            for i in range(meas.nchannels):
                self.channel.addItem(str(i))
            self.channel.setCurrentIndex(0)

    def compute(self):
        """
        Compute Sound Level using settings. This method is
        called whenever the Compute button is pushed in the SLM tab
        """
        meas = self.meas
        fs = meas.samplerate
        channel = self.channel.currentIndex()
        tw = TimeWeighting.getCurrent(self.timeweighting)
        fw = FreqWeighting.getCurrent(self.freqweighting)

        # Downsampling factor of result
        dsf = self.downsampling.value()
        # gb = self.slmFre

        with wait_cursor():
            # This one exctracts the calfile and sensitivity from global
            # variables defined at the top. # TODO: Change this to a more
            # robust variant.
            weighcal = WeighCal(fw, nchannels=1,
                                fs=fs, calfile=calfile,
                                sens=sens)

            slm = SLM(fs, weighcal, tw)
            praw = meas.praw()[:, [channel]]

            # Filter, downsample data
            filtered = slm.addData(praw)[::dsf, :]
            N = filtered.shape[0]
            time = getTime(float(fs)/dsf, N)
            Lmax = slm.Lmax

            pto = PlotOptions()
            pto.ylabel = f'L{fw[0]} [dB({fw[0]})]'
            pto.xlim = (time[0], time[-1])
            fig, new = self.getFigure(pto)
            fig.fig.plot(time, filtered)
            fig.show()

        stats = f"""Statistical results:
=============================
Applied frequency weighting: {fw[1]}
Applied time weighting: {tw[1]}
Applied Downsampling factor: {dsf}
Maximum level (L{fw[0]} max): {Lmax:4.4} [dB({fw[0]})]

        """
        self.results.setPlainText(stats)
