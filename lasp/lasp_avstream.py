#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Description: Read data from image stream and record sound at the same time
"""
import cv2 as cv
from .lasp_atomic import Atomic
from threading import Thread, Condition, Lock
import numpy as np

import time
from .device import (RtAudio, DeviceInfo, DAQConfiguration,
                     get_numpy_dtype_from_format_string,
                     get_sampwidth_from_format_string)

__all__ = ['AvType', 'AvStream']

video_x, video_y = 640, 480


class AvType:
    """Specificying the type of data, for adding and removing callbacks from
    the stream."""
    audio_input = 1
    audio_output = 2
    video = 4


class AvStream:
    """Audio and video data stream, to which callbacks can be added for
    processing the data."""

    def __init__(self,
                 device: DeviceInfo,
                 avtype: AvType,
                 daqconfig: DAQConfiguration,
                 video=None):
        """Open a stream for audio in/output and video input. For audio output,
        by default all available channels are opened for outputting data.

        Args:
            device: DeviceInfo for the audio device
            avtype: Type of stream. Input, output or duplex

            daqconfig: DAQConfiguration instance. If duplex mode flag is set,
            please make sure that output_device is None, as in that case the
            output config will be taken from the input device.
            video:
        """

        self.daqconfig = daqconfig
        self._device = device
        self.avtype = avtype
        self.duplex_mode = daqconfig.duplex_mode
        self.monitor_gen = daqconfig.monitor_gen

        # Determine highest input channel number
        channelconfigs = daqconfig.en_input_channels

        self.channel_names = []
        self.sensitivity = self.daqconfig.getSensitivities()
        if daqconfig.monitor_gen:
            assert self.duplex_mode
            self.channel_names.append('Generated signal')
            self.sensitivity = np.concatenate([np.array([1.]),
                                               self.sensitivity])

        rtaudio_inputparams = None
        rtaudio_outputparams = None

        self.nframes_per_block = 2048

        if self.duplex_mode or avtype == AvType.audio_output:
            rtaudio_outputparams = {'deviceid': device.index,
                                    # TODO: Add option to specify the number of output channels to use
                                    'nchannels': 1,  # device.outputchannels,
                                    'firstchannel': 0}
            self.sampleformat = daqconfig.en_output_sample_format
            self.samplerate = int(daqconfig.en_output_rate)

        if avtype == AvType.audio_input or self.duplex_mode:
            for i, channelconfig in enumerate(channelconfigs):
                if channelconfig.channel_enabled:
                    self.nchannels = i+1
                    self.channel_names.append(channelconfig.channel_name)
            rtaudio_inputparams = {'deviceid': device.index,
                                   'nchannels': self.nchannels,
                                   'firstchannel': firstchannel}

            # Here, we override the sample format in case of duplex mode.
            self.sampleformat = daqconfig.en_input_sample_format
            self.samplerate = int(daqconfig.en_input_rate)

        try:
            self._rtaudio = RtAudio()
            self.blocksize = self._rtaudio.openStream(
                rtaudio_outputparams,  # Outputparams
                rtaudio_inputparams,   # Inputparams
                self.sampleformat,     # Sampleformat
                self.samplerate,
                self.nframes_per_block,  # Buffer size in frames
                self._audioCallback)

        except Exception as e:
            raise RuntimeError(f'Could not initialize DAQ device: {str(e)}')

        # Fill in numpy data type, and sample width
        self.numpy_dtype = get_numpy_dtype_from_format_string(
            self.sampleformat)
        self.sampwidth = get_sampwidth_from_format_string(
            self.sampleformat)

        # Counters for the number of frames that have been coming in
        self._aframectr = Atomic(0)
        self._vframectr = Atomic(0)

        # Lock
        self._callbacklock = Lock()

        self._running = Atomic(False)
        self._running_cond = Condition()

        self._video = video
        self._video_started = Atomic(False)

        # Storage for callbacks, specified by type
        self._callbacks = {
            AvType.audio_input: [],
            AvType.audio_output: [],
            AvType.video: []
        }

        # Possible, but long not tested: store video
        self._videothread = None

    def close(self):
        self._rtaudio.closeStream()
        self._rtaudio = None

    def nCallbacks(self):
        """Returns the current number of installed callbacks."""
        return len(self._callbacks[AvType.audio_input]) + \
            len(self._callbacks[AvType.audio_output]) + \
            len(self._callbacks[AvType.video])

    def addCallback(self, cb: callable, cbtype: AvType):
        """Add as stream callback to the list of callbacks."""
        with self._callbacklock:
            outputcallbacks = self._callbacks[AvType.audio_output]
            if cbtype == AvType.audio_output and len(outputcallbacks) > 0:
                raise RuntimeError(
                    'Only one audio output callback can be allowed')

            if cb not in self._callbacks[cbtype]:
                self._callbacks[cbtype].append(cb)

    def removeCallback(self, cb, cbtype: AvType):
        with self._callbacklock:
            if cb in self._callbacks[cbtype]:
                self._callbacks[cbtype].remove(cb)

    def start(self):
        """Start the stream, which means the callbacks are called with stream
        data (audio/video)"""

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
                    for cb in self._callbacks[AvType.video]:
                        cb(frame, vframectr)
                vframectr += 1
                self._vframectr += 1
            else:
                loopctr += 1
                if loopctr == 10:
                    print('Error: no video capture!')
            time.sleep(0.2)

        cap.release()
        print('stopped videothread')

    def _audioCallback(self, indata, outdata, nframes, streamtime):
        """This is called (from a separate thread) for each audio block."""
        self._aframectr += nframes
        with self._callbacklock:

            # Count the number of output callbacks. If no output callbacks are
            # present, and there should be output callbacks, we explicitly set
            # the output buffer to zero
            noutput_cb = len(self._callbacks[AvType.audio_output])
            shouldhaveoutput = (self.avtype == AvType.audio_output or
                                self.duplex_mode)
            if noutput_cb == 0 and shouldhaveoutput:
                outdata[:, :] = 0

            # Loop over callbacks
            for cb in self._callbacks[AvType.audio_output]:
                try:
                    cb(indata, outdata, self._aframectr())
                except Exception as e:
                    print(e)
                    return 1
            for cb in self._callbacks[AvType.audio_input]:
                try:
                    cb(indata, outdata, self._aframectr())
                except Exception as e:
                    print(e)
                    return 1

        return 0 if self._running else 1

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
        self._rtaudio.stopStream()

    def isRunning(self):
        return self._running()

    def hasVideo(self):
        return True if self._video is not None else False
