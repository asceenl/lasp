#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Read data from image stream and record sound at the same time
"""
import cv2 as cv
from .lasp_atomic import Atomic
from threading import Thread, Condition, Lock
import time
from .device import DAQConfiguration
import numpy as np
__all__ = ['AvType', 'AvStream']

video_x, video_y = 640, 480
dtype, sampwidth = 'int16', 2


class AvType:
    video = 0
    audio = 1


class AvStream:
    def __init__(self,
                 rtaudio,
                 output_device,
                 input_device,
                 daqconfig, video=None):

        self.daqconfig = daqconfig
        self.input_device = input_device
        self.output_device = output_device

        # Determine highest input channel number
        channelconfigs = daqconfig.en_input_channels
        max_input_channel = 0

        self._rtaudio = rtaudio
        self.samplerate = int(daqconfig.en_input_rate)
        self.sensitivity = self.daqconfig.getSensitivities()

        for i, channelconfig in enumerate(channelconfigs):
            if channelconfig.channel_enabled:
                self.nchannels = i+1

        try:
            if input_device is not None:
                inputparams = {'deviceid': input_device.index,
                               'nchannels': self.nchannels,
                               'firstchannel': 0}

                self.blocksize = rtaudio.openStream(
                        None, # Outputparams
                        inputparams, #Inputparams
                        daqconfig.en_input_sample_format, # Sampleformat
                        self.samplerate,
                        2048,
                        self._audioCallback)

        except Exception as e:
            raise RuntimeError(f'Could not initialize DAQ device: {str(e)}')

        self.video_x, self.video_y = video_x, video_y
        self.dtype, self.sampwidth = dtype, sampwidth

        self._aframectr = Atomic(0)
        self._vframectr = Atomic(0)

        self._callbacklock = Lock()

        self._running = Atomic(False)
        self._running_cond = Condition()

        self._video = video
        self._video_started = Atomic(False)
        self._callbacks = []
        self._videothread = None

    def close(self):
        self._rtaudio.closeStream()

    def nCallbacks(self):
        """
        Returns the current number of installed callbacks
        """
        return len(self._callbacks)

    def addCallback(self, cb):
        """
        Add as stream callback to the list of callbacks
        """
        with self._callbacklock:
            if cb not in self._callbacks:
                self._callbacks.append(cb)

    def removeCallback(self, cb):
        with self._callbacklock:
            if cb in self._callbacks:
                self._callbacks.remove(cb)

    def start(self):
        """
        Start the stream, which means the callbacks are called with stream
        data (audio/video)
        """

        if self._running:
            raise RuntimeError('Stream already started')

        assert self._videothread is None

        self._running <<= True
        if self._video is not None:
            self._videothread = Thread(target=self._videoThread)
            self._videothread.start()
        else:
            self._video_started <<= True
        self._rtaudio.startStream()

    def _videoThread(self):
        cap = cv.VideoCapture(self._video)
        if not cap.isOpened():
            cap.open()
        vframectr = 0
        loopctr = 0
        while self._running:
            ret, frame = cap.read()
            # print(frame.shape)
            if ret is True:
                if vframectr == 0:
                    self._video_started <<= True
                with self._callbacklock:
                    for cb in self._callbacks:
                        cb(AvType.video, frame, self._aframectr(), vframectr)
                vframectr += 1
                self._vframectr += 1
            else:

                if loopctr == 10:
                    print('Error: no video capture!')
            time.sleep(0.2)
            loopctr += 1

        cap.release()
        print('stopped videothread')

    def _audioCallback(self, indata, nframes, streamtime):
        """
        This is called (from a separate thread) for each audio block.
        """
        self._aframectr += 1
        with self._callbacklock:
            for cb in self._callbacks:
                cb(AvType.audio, indata, self._aframectr(), self._vframectr())
        return None, 0 if self._running else 1

    def stop(self):
        self._running <<= False
        with self._running_cond:
            self._running_cond.notify()
        if self._video:
            self._videothread.join()
            self._videothread = None
        self._aframectr <<= 0
        self._vframectr <<= 0
        self._video_started <<= False

    def isRunning(self):
        return self._running()

    def hasVideo(self):
        return True if self._video is not None else False
