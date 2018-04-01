#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 08:28:03 2018

@author: Read data from image stream and record sound at the same time
"""
import cv2 as cv
import numpy as np
import queue
import sounddevice as sd
import time
from matplotlib.pyplot import plot, show
from .lasp_atomic import Atomic
from threading import Thread, Condition
import h5py

class Playback:
    """
    Play back a single channel from a 
    """
    def __init__(self, fn1, channel = 0, video=True, verbose=True):
        """
        
        """
        ext = '.h5'
        if not ext in fn1:
            fn = fn1 +  ext
        else:
            fn = fn1

        print('Filename: ', fn)
        self._fn = fn
        
        self.channel = channel
        self._video = video
        self._aframectr = Atomic(0)
        self._running = Atomic(False)
        self._running_cond = Condition()
        if video:
            self._video_queue = queue.Queue()

        with h5py.File(fn,'r') as f:
            self.samplerate = f.attrs['samplerate']
            self.nchannels = f.attrs['nchannels']
            self.blocksize = f.attrs['blocksize']
            self.nblocks = f['audio'].shape[0]
            if verbose:
                print('Sample rate: ', self.samplerate)
                print('Number of audio frames: ', self.nblocks*self.blocksize)
                print('Recording time: ', self.nblocks*self.blocksize/self.samplerate)
            
            
            if video:
                try:
                    f['video']
                    self._video_frame_positions = f['video_frame_positions'][:]
                except:
                    print('No video available in measurement file. Disabling video')
                    self._video = False
            
    @property
    def T(self):
        return self._nblocks*self._blocksize/self._samplerate
    
    def start(self):
        with h5py.File(self._fn,'r') as f:
            stream = sd.OutputStream(samplerate = self.samplerate,
                                     blocksize = self.blocksize,
                                     channels = 1,
                                     dtype = 'int32',callback = self.audio_callback)
            self._ad = f['audio']
            self._running <<= True
            if self._video:
                self._vd = f['video']
                videothread = Thread(target=self.video_thread_fcn)
                videothread.start()
                
            with stream:
                try:
                    with self._running_cond:
                        while self._running:
                            self._running_cond.wait()
                except KeyboardInterrupt:
                    print('Keyboard interrupt. Quit playback')
                
            if self._video:
                videothread.join()
    
    def audio_callback(self, outdata, frames, time, status):
        """
        
        """
        aframectr = self._aframectr()
        if aframectr < self.nblocks:
            outdata[:,0] = self._ad[aframectr,:,self.channel]
            self._aframectr += 1
        else:
            self._running <<= False
            with self._running_cond:
                self._running_cond.notify()
        
    def video_thread_fcn(self):
        frame_ctr = 0
        nframes = self._vd.shape[0]
        video_frame_positions = self._video_frame_positions
        assert video_frame_positions.shape[0] == nframes
        
        while self._running and frame_ctr < nframes:
            frame = self._vd[frame_ctr]
            
            # Find corresponding audio frame
            corsp_aframe = video_frame_positions[frame_ctr]
            
            while self._aframectr() <= corsp_aframe:
                print('Sleep video...')
                time.sleep(self.blocksize/self.samplerate/2)

            cv.imshow("Video output. Press 'q' to quit",frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                self._running <<= False
            
            frame_ctr += 1
        print('Ending video playback thread')
        cv.destroyAllWindows()
    
