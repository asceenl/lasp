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
from .lasp_figure import PlotOptions, Plotable
from .ui_slmwidget import Ui_SlmWidget
from .filter.bandpass_fir import OctaveBankDesigner, ThirdOctaveBankDesigner
from .lasp_octavefilter import OctaveFilterBank, ThirdOctaveFilterBank
__all__ = ['SLM', 'SlmWidget']


class Dummy:
    """
    Emulate filtering, but does not filter anything at all.
    """

    def filter_(self, data):
        return data[:, np.newaxis]


class SLM:
    """
    Multi-channel sound Level Meter. Input data: time data with a certain
    sampling frequency. Output: time-weighted (fast/slow) sound pressure
    levels in dB(A/C/Z).

    """

    def __init__(self, fs, tw=TimeWeighting.default):
        """
        Initialize a sound level meter object.
        Number of channels comes from the weighcal object.

        Args:
            fs: Sampling frequency [Hz]
            tw: Time Weighting to apply
        """

        if tw[0] is not TimeWeighting.none[0]:
            self._lp = SPLowpass(fs, tw[0])
        else:
            self._lp = Dummy()
        self._Lmax = 0.

        # Storage for computing the equivalent level
        self._sq = 0.
        self._N = 0
        self._Leq = 0.

    @property
    def Leq(self):
        """
        Returns the equivalent level of the recorded signal so far
        """
        return self._Leq

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
        assert data.ndim == 2
        assert data.shape[1] == 1

        P_REFsq = P_REF**2

        # Squared
        sq = data**2

        # Update equivalent level
        N1 = sq.shape[0]
        self._sq = (np.sum(sq) + self._sq*self._N)/(N1+self._N)
        self._N += N1
        self._Leq = 10*np.log10(self._sq/P_REFsq)

        # Time-weight the signal
        tw = self._lp.filter_(sq)

        Level = 10*np.log10(tw/P_REFsq)

        # Update maximum level
        curmax = np.max(Level)
        if curmax > self._Lmax:
            self._Lmax = curmax

        return Level


class SlmWidget(ComputeWidget, Ui_SlmWidget):
    def __init__(self, parent=None):
        """
        Initialize the SlmWidget.
        """
        super().__init__(parent)
        self.setupUi(self)

        self.eqFreqBandChanged(0)
        self.tFreqBandChanged(0)
        self.setMeas(None)

    def init(self, fm):
        """
        Register combobox of the figure dialog to plot to in the FigureManager
        """
        super().init(fm)
        fm.registerCombo(self.tfigure)
        fm.registerCombo(self.eqfigure)

        self.tbandstart.setEnabled(False)
        self.tbandstop.setEnabled(False)

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
            self.tstarttime.setRange(0, rt, 0)
            self.tstoptime.setRange(0, rt, rt)

            self.eqstarttime.setRange(0, rt, 0)
            self.eqstoptime.setRange(0, rt, rt)

            self.tchannel.clear()
            self.eqchannel.clear()
            for i in range(meas.nchannels):
                self.tchannel.addItem(str(i))
                self.eqchannel.addItem(str(i))
            self.tchannel.setCurrentIndex(0)
            self.eqchannel.setCurrentIndex(0)

    def computeEq(self):
        """
        Compute equivalent levels for a piece of time
        """
        meas = self.meas
        fs = meas.samplerate
        channel = self.eqchannel.currentIndex()
        fw = FreqWeighting.getCurrent(self.eqfreqweighting)

        istart, istop = self.getStartStopIndices(meas, self.eqstarttime,
                                                 self.eqstoptime)

        bands = self.eqfreqband.currentIndex()
        if bands == 0:
            # 1/3 Octave bands
            filt = ThirdOctaveFilterBank(fs)
            xs = filt.xs
            xmin = xs[0] + self.eqbandstart.currentIndex()
            xmax = xs[0] + self.eqbandstop.currentIndex()
        if bands == 1:
            # Octave bands
            filt = OctaveFilterBank(fs)
            xs = filt.xs
            xmin = xs[0] + self.eqbandstart.currentIndex()
            xmax = xs[0] + self.eqbandstop.currentIndex()

        leveltype = self.eqleveltype.currentIndex()
        if leveltype == 0:
            # equivalent levels
            tw = TimeWeighting.fast
        elif leveltype == 1:
            # fast time weighting
            tw = TimeWeighting.fast
        elif leveltype == 2:
            # slow time weighting
            tw = TimeWeighting.slow

        with wait_cursor():
            # This one exctracts the calfile and sensitivity from global
            # variables defined at the top. # TODO: Change this to a more
            # robust variant.
            weighcal = WeighCal(fw, nchannels=1,
                                fs=fs, calfile=calfile,
                                sens=sens)
            praw = meas.praw()[istart:istop, [channel]]

            weighted = weighcal.filter_(praw)
            filtered_out = filt.filter_(weighted)

            levels = np.empty((xmax - xmin + 1))
            xlabels = []
            for i, x in enumerate(range(xmin, xmax+1)):
                nom = filt.nominal(x)
                xlabels.append(nom)
                filt_x = filtered_out[nom]['data']
                slm = SLM(filt.fs, tw)
                slm.addData(filt_x)
                if leveltype > 0:
                    level = slm.Lmax
                else:
                    level = slm.Leq
                levels[i] = level

            pto = PlotOptions.forLevelBars()
            pta = Plotable(xlabels, levels)
            fig, new = self.getFigure(self.eqfigure, pto, 'bar')
            fig.fig.add(pta)
            fig.show()

    def computeT(self):
        """
        Compute sound levels as a function of time.
        """
        meas = self.meas
        fs = meas.samplerate
        channel = self.tchannel.currentIndex()
        tw = TimeWeighting.getCurrent(self.ttimeweighting)
        fw = FreqWeighting.getCurrent(self.tfreqweighting)

        istart, istop = self.getStartStopIndices(meas, self.tstarttime,
                                                 self.tstoptime)

        bands = self.tfreqband.currentIndex()
        if bands == 0:
            # Overall
            filt = Dummy()
        else:
            # Octave bands
            filt = OctaveFilterBank(
                fs) if bands == 1 else ThirdOctaveFilterBank(fs)
            xs = filt.xs
            xmin = xs[0] + self.tbandstart.currentIndex()
            xmax = xs[0] + self.tbandstop.currentIndex()

        # Downsampling factor of result
        dsf = self.tdownsampling.value()

        with wait_cursor():
            # This one exctracts the calfile and sensitivity from global
            # variables defined at the top. # TODO: Change this to a more
            # robust variant.

            praw = meas.praw()[istart:istop, [channel]]

            weighcal = WeighCal(fw, nchannels=1,
                                fs=fs, calfile=calfile,
                                sens=sens)

            weighted = weighcal.filter_(praw)

            if bands == 0:
                slm = SLM(fs, tw)
                level = slm.addData(weighted)[::dsf]

                # Filter, downsample data
                N = level.shape[0]
                time = getTime(float(fs)/dsf, N)
                Lmax = slm.Lmax

                pta = Plotable(time, level,
                               name=f'Overall level [dB([fw[0]])]')
                pto = PlotOptions()
                pto.ylabel = f'L{fw[0]} [dB({fw[0]})]'
                pto.xlim = (time[0], time[-1])
                fig, new = self.getFigure(self.tfigure, pto, 'line')
                fig.fig.add(pta)

            else:
                pto = PlotOptions()
                fig, new = self.getFigure(self.tfigure, pto, 'line')
                pto.ylabel = f'L{fw[0]} [dB({fw[0]})]'

                out = filt.filter_(weighted)
                tmin = 0
                tmax = 0

                for x in range(xmin, xmax+1):
                    dec = np.prod(filt.decimation(x))
                    fd = filt.fs/dec
                    # Nominal frequency text
                    nom = filt.nominal(x)

                    leg = f'{nom} Hz - [dB({fw[0]})]'

                    # Find global tmin and tmax, used for xlim
                    time = out[nom]['t']
                    tmin = min(tmin, time[0])
                    tmax = max(tmax, time[-1])
                    slm = SLM(fd, tw)
                    level = slm.addData(out[nom]['data'])
                    plotable = Plotable(time[::dsf//dec],
                                        level[::dsf//dec],
                                        name=leg)

                    fig.fig.add(plotable)
                pto.xlim = (tmin, tmax)
                fig.fig.setPlotOptions(pto)
            fig.show()

#         stats = f"""Statistical results:
# =============================
# Applied frequency weighting: {fw[1]}
# Applied time weighting: {tw[1]}
# Applied Downsampling factor: {dsf}
# Maximum level (L{fw[0]} max): {Lmax:4.4} [dB({fw[0]})]
#
#         """
#         self.results.setPlainText(stats)

    def compute(self):
        """
        Compute Sound Level using settings. This method is
        called whenever the Compute button is pushed in the SLM tab
        """
        if self.ttab.isVisible():
            self.computeT()
        elif self.eqtab.isVisible():
            self.computeEq()

    def eqFreqBandChanged(self, idx):
        """
        User changes frequency bands to plot time-dependent values for
        """
        self.eqbandstart.clear()
        self.eqbandstop.clear()

        if idx == 1:
            # 1/3 Octave bands
            o = OctaveBankDesigner()
            for x in o.xs:
                nom = o.nominal(x)
                self.eqbandstart.addItem(nom)
                self.eqbandstop.addItem(nom)
                self.eqbandstart.setCurrentIndex(0)
                self.eqbandstop.setCurrentIndex(len(o.xs)-1)
        elif idx == 0:
            # Octave bands
            o = ThirdOctaveBankDesigner()
            for x in o.xs:
                nom = o.nominal(x)
                self.eqbandstart.addItem(nom)
                self.eqbandstop.addItem(nom)
                self.eqbandstart.setCurrentIndex(2)
                self.eqbandstop.setCurrentIndex(len(o.xs) - 3)

    def tFreqBandChanged(self, idx):
        """
        User changes frequency bands to plot time-dependent values for
        """
        self.tbandstart.clear()
        self.tbandstop.clear()
        enabled = False

        if idx == 1:
            # Octave bands
            enabled = True
            o = OctaveBankDesigner()
            for x in o.xs:
                nom = o.nominal(x)
                self.tbandstart.addItem(nom)
                self.tbandstop.addItem(nom)
                self.tbandstart.setCurrentIndex(2)
                self.tbandstop.setCurrentIndex(len(o.xs)-1)
        elif idx == 2:
            # Octave bands
            enabled = True
            o = ThirdOctaveBankDesigner()
            for x in o.xs:
                nom = o.nominal(x)
                self.tbandstart.addItem(nom)
                self.tbandstop.addItem(nom)
                self.tbandstart.setCurrentIndex(2)
                self.tbandstop.setCurrentIndex(len(o.xs) - 3)

        self.tbandstart.setEnabled(enabled)
        self.tbandstop.setEnabled(enabled)
