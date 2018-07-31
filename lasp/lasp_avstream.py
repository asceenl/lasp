#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 08:28:03 2018

@author: Read data from image stream and record sound at the same time
"""
import cv2 as cv
from .lasp_atomic import Atomic
from threading import Thread, Condition, Lock
import time
from .device import DAQDevice, roga_plugndaq
__all__ = ['AvType', 'AvStream']

video_x, video_y = 640, 480
dtype, sampwidth = 'int16', 2


class AvType:
    video = 0
    audio = 1


class AvStream:
    def __init__(self, daqconfig=roga_plugndaq, video=None):

        self.daqconfig = daqconfig
        try:
            daq = DAQDevice(daqconfig)
            self.nchannels = len(daq.channels_en)
            self.samplerate = daq.input_rate
            self.blocksize = daq.blocksize
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
        self._audiothread = None
        self._videothread = None

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

        assert self._audiothread is None
        assert self._videothread is None

        self._running <<= True
        self._audiothread = Thread(target=self._audioThread)
        if self._video is not None:
            self._videothread = Thread(target=self._videoThread)
            self._videothread.start()
        else:
            self._video_started <<= True
        self._audiothread.start()

    def _audioThread(self):
        # Raw stream to allow for in24 packed data type
        try:
            daq = DAQDevice(self.daqconfig)
            # Get a single block first and do not process it. This one often
            # contains quite some rubbish.
            data = daq.read()
            while self._running:
                data = daq.read()
                self._audioCallback(data)
        except RuntimeError as e:
            print(f'Runtime error occured during audio capture: {str(e)}')

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

    def _audioCallback(self, indata):
        """This is called (from a separate thread) for each audio block."""
        if not self._video_started:
            return

        with self._callbacklock:
            for cb in self._callbacks:
                cb(AvType.audio, indata, self._aframectr(), self._vframectr())
        self._aframectr += 1

    def stop(self):
        self._running <<= False
        with self._running_cond:
            self._running_cond.notify()
        self._audiothread.join()
        self._audiothread = None
        if self._video:
            self._videothread.join()
            self._videothread = None
        self._aframectr <<= 0
        self._vframectr <<= 0
        self._video_started <<= False

    def isStarted(self):
        return self._running()

    def hasVideo(self):
        return True if self._video is not None else False
