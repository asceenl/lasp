#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Description: Read data from image stream and record sound at the same time
"""
from .lasp_atomic import Atomic
from threading import Thread, Condition, Lock
from .lasp_avstream import AvType, AvStream
import numpy as np

class SignalGenerator:
    """
    Base class for all signal generator types
    """

    def __init__(self, stream):

        if not (stream.avtype == AvType.audio_output or (stream.avtype ==
                                                        AvType.audio_input and
                                                        stream.duplex_mode)):
            raise RuntimeError('Invalid stream type. Does not support audio'
                               'output')

        self.stream = stream
        self._samplerate = stream.samplerate
        stream.addCallback(self.streamCallback, AvType.audio_output)

    def stop(self):
        """
        Stop generating the signal
        """
        self.stream.removeCallback(self.streamCallback, AvType.audio_output)

    def streamCallback(self, indata, outdata, blockctr):
        """
        Callback from AvStream.
        """
        signal = self.getSignal(blockctr*outdata.shape[0], outdata.shape[0])
        dtype = self.stream.numpy_dtype

        if dtype == np.float32 or dtype == np.float64:
            fac = 1
        else:
            bitdepth_fixed = self.stream.sampwidth*8
            fac = 2**(bitdepth_fixed-1)

        outdata[:, :] = (fac*signal).astype(dtype)[:, np.newaxis]


class SineGenerator(SignalGenerator):
    def __init__(self, stream, freq):
        self._omg = 2*freq*np.pi
        super().__init__(stream)

    def getSignal(self, startidx, frames):
        samplerate = self._samplerate
        streamtime = startidx/samplerate
        t = np.linspace(streamtime, streamtime + frames/samplerate, frames)

        return 0.1*np.sin(self._omg*t)


class NoiseGenerator(SignalGenerator):
    def __init__(self, stream, noisetype): 
        super().__init__(stream)
    
    def getSignal(self, startidx, frames):
        return 0.1*np.random.randn(frames)

class NoiseType:
    white = (0, 'White noise', )
    pink = (1, 'Pink noise')
    types = (white, pink)

    @staticmethod
    def fillComboBox(combo):
        for type_ in NoiseType.types:
            combo.addItem(type_[1])

    @staticmethod
    def getCurrent(cb):
        return NoiseType.types[cb.currentIndex()]




