#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 08:28:03 2018

Read data from stream and record sound and video at the same time
"""
import numpy as np
from .lasp_atomic import Atomic
from threading import Condition
from .lasp_avstream import AvType, AvStream
import h5py
import time


class Recording:
    def __init__(self, fn, stream, rectime=None):
        """

        Args:
            fn: Filename to record to. extension is added
            stream: AvStream instance to record from
            rectime: Recording time, None for infinite
        """
        ext = '.h5'
        if ext not in fn:
            fn += ext
        self._stream = stream
        self.blocksize = stream.blocksize
        self.samplerate = stream.samplerate
        self._running = Atomic(False)
        self._running_cond = Condition()
        self.rectime = rectime
        self._fn = fn

        self._video_frame_positions = []

        self._aframeno = Atomic(0)
        self._vframeno = 0

    def start(self):
        stream = self._stream

        with h5py.File(self._fn, 'w') as f:
            self._ad = f.create_dataset('audio',
                                        (1, stream.blocksize, stream.nchannels),
                                        dtype=stream.dtype,
                                        maxshape=(None, stream.blocksize,
                                                  stream.nchannels),
                                        compression='gzip'
                                        )
            if stream.hasVideo():
                video_x, video_y = stream.video_x, stream.video_y
                self._vd = f.create_dataset('video',
                                            (1, video_y, video_x, 3),
                                            dtype='uint8',
                                            maxshape=(
                                                None, video_y, video_x, 3),
                                            compression='gzip'
                                            )

            f.attrs['samplerate'] = stream.samplerate
            f.attrs['nchannels'] = stream.nchannels
            f.attrs['blocksize'] = stream.blocksize
            f.attrs['sensitivity'] = stream.sensitivity
            f.attrs['time'] = time.time()
            self._running <<= True
            # Videothread is going to start

            if not stream.isStarted():
                stream.start()

            stream.addCallback(self._callback)
            with self._running_cond:
                try:
                    print('Starting record....')
                    while self._running:
                        self._running_cond.wait()
                except KeyboardInterrupt:
                    print("Keyboard interrupt on record")
                    self._running <<= False

            stream.removeCallback(self._callback)

            if stream.hasVideo():
                f['video_frame_positions'] = self._video_frame_positions

            print('\nEnding record')

    def stop(self):
        self._running <<= False
        with self._running_cond:
            self._running_cond.notify()

    def _callback(self, _type, data, aframe, vframe):
        if not self._stream.isStarted():
            self._running <<= False
            with self._running_cond:
                self._running_cond.notify()

        if _type == AvType.audio:
            self._aCallback(data, aframe)
        elif _type == AvType.video:
            self._vCallback(data)

    def _aCallback(self, frames, aframe):
        print('.', end='')
        curT = self._aframeno()*self.blocksize/self.samplerate
        if self.rectime is not None and curT > self.rectime:
            # We are done!
            self._running <<= False
            with self._running_cond:
                self._running_cond.notify()
                return

        self._ad.resize(self._aframeno()+1, axis=0)
        self._ad[self._aframeno(), :, :] = frames
        self._aframeno += 1

    def _vCallback(self, frame):
        self._video_frame_positions.append(self._aframeno())
        vframeno = self._vframeno
        self._vd.resize(vframeno+1, axis=0)
        self._vd[vframeno, :, :] = frame
        self._vframeno += 1


if __name__ == '__main__':
    stream = AvStream()
    rec = Recording('test', stream, 5)
    rec.start()
