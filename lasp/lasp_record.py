#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Read data from stream and record sound and video at the same time
"""
from .lasp_atomic import Atomic
from threading import Condition
from .lasp_avstream import AvType, AvStream
import h5py
import dataclasses
import os
import time

@dataclasses.dataclass
class RecordStatus:
    curT: float
    done: bool

class Recording:

    def __init__(self, fn: str, stream: AvStream, 
                 rectime: float=None, wait: bool = True,
                 progressCallback=None):
        """

        Args:
            fn: Filename to record to. extension is added
            stream: AvStream instance to record from. Should have input
            channels!
            rectime: Recording time, None for infinite
        """
        ext = '.h5'
        if ext not in fn:
            fn += ext

        self._stream = stream
        if stream.avtype != AvType.audio_input:
            raise RuntimeError('Stream does not have any input channels')
        self.blocksize = stream.blocksize
        self.samplerate = stream.samplerate
        self._running = Atomic(False)
        self._running_cond = Condition()
        self.rectime = rectime
        self._fn = fn

        self._video_frame_positions = []
        self._curT_rounded_to_seconds = 0

        self._ablockno = Atomic(0)
        self._vframeno = 0

        self._progressCallback = progressCallback 
        self._wait = wait

        self._f = h5py.File(self._fn, 'w')
        self._deleteFile = False

    def setDelete(self, val: bool):
        """
        Set the delete flag. If set, measurement file is deleted at the end of
        the recording. Typically used for cleaning up after canceling a
        recording.
        """
        self._deleteFile = val

    def __enter__(self):
        """

        with Recording(fn, stream, wait=False):
            event_loop_here()

        or:

        with Recording(fn, stream, wait=True):
            pass
        """

        stream = self._stream
        f = self._f
        nchannels = stream.nchannels
        if stream.monitor_gen:
            nchannels += 1

        self.monitor_gen = stream.monitor_gen

        self._ad = f.create_dataset('audio',
                                    (1, stream.blocksize, nchannels),
                                    dtype=stream.numpy_dtype,
                                    maxshape=(None, stream.blocksize,
                                              nchannels),
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
        f.attrs['channel_names'] = stream.channel_names
        f.attrs['time'] = time.time()
        self._running <<= True

        if not stream.isRunning():
            stream.start()

        print('Starting record....')
        stream.addCallback(self._aCallback, AvType.audio_input)
        if stream.hasVideo():
            stream.addCallback(self._aCallback, AvType.audio_input)

        if self._wait:
            with self._running_cond:
                print('Stop recording with CTRL-C')
                try:
                    while self._running:
                        self._running_cond.wait()
                except KeyboardInterrupt:
                    print("Keyboard interrupt on record")
                    self._running <<= False


    def __exit__(self, type, value, traceback):
        self._running <<= False
        stream = self._stream
        stream.removeCallback(self._aCallback, AvType.audio_input)
        if stream.hasVideo():
            stream.removeCallback(self._vCallback, AvType.video_input)
            self._f['video_frame_positions'] = self._video_frame_positions

        self._f.close()
        print('\nEnding record')
        if self._deleteFile:
            try:
                os.remove(self._fn)
            except Exception as e:
                print(f'Error deleting file: {self._fn}')


    def _aCallback(self, indata, outdata, aframe):

        curT = self._ablockno()*self.blocksize/self.samplerate
        recstatus = RecordStatus(
            curT = curT,
            done = False)
        if self._progressCallback is not None:
            self._progressCallback(recstatus)

        curT_rounded_to_seconds = int(curT)
        if curT_rounded_to_seconds > self._curT_rounded_to_seconds:
            self._curT_rounded_to_seconds = curT_rounded_to_seconds
            print(f'{curT_rounded_to_seconds}', end='', flush=True)
        else:
            print('.', end='', flush=True)

        if self.rectime is not None and curT > self.rectime:
            # We are done!
            self._running <<= False
            with self._running_cond:
                self._running_cond.notify()
            if self._progressCallback is not None:
                recstatus.done = True
                self._progressCallback(recstatus)
            return

        self._ad.resize(self._ablockno()+1, axis=0)
        if self.monitor_gen:
            self._ad[self._ablockno(), :, 0] = outdata[:, 0]
            self._ad[self._ablockno(), :, 1:] = indata
        else:
            self._ad[self._ablockno(), :, :] = indata
        self._ablockno += 1

    def _vCallback(self, frame, framectr):
        self._video_frame_positions.append(self._ablockno())
        vframeno = self._vframeno
        self._vd.resize(vframeno+1, axis=0)
        self._vd[vframeno, :, :] = frame
        self._vframeno += 1

