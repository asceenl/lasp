#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 08:28:03 2018

@author: Read data from image stream and record sound at the same time
"""
import cv2 as cv
import sounddevice as sd
from .lasp_atomic import Atomic
from threading import Thread, Condition, Lock
import time
__all__ = ['AvType','AvStream']

# %%
blocksize = 2048
video_x,video_y = 640,480
dtype, sampwidth = 'int32',4

class AvType:
    video=0
    audio=1

class AvStream:
    def __init__(self, audiodeviceno=None, video=None, nchannels = None, samplerate = None):
        
        audiodevice,audiodeviceno = self._findDevice(audiodeviceno)
        if nchannels is None:
            self.nchannels = audiodevice['max_input_channels']
            if self.nchannels == 0:
                raise RuntimeError('Device has no input channels')
        else:
            self.nchannels = nchannels
        
        self.audiodeviceno = audiodeviceno
        if samplerate is None:
            self.samplerate = audiodevice['default_samplerate']
        else:
            self.samplerate = samplerate
            
        self.blocksize = blocksize
        
        self.video_x, self.video_y = video_x,video_y
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
        
        """
        with self._callbacklock:
            if not cb in self._callbacks:
                self._callbacks.append(cb)
                
    def removeCallback(self, cb):
        with self._callbacklock:
            if cb in self._callbacks:
                self._callbacks.remove(cb)

    def _findDevice(self, deviceno):
        
        if deviceno is None:
            deviceno = 0
            devices = sd.query_devices()
            found = []
            for device in devices:
                name = device['name']
                if 'Umik' in name:
                    found.append((device,deviceno))
                elif 'nanoSHARC' in name:
                    found.append((device,deviceno))
                deviceno+=1
                
            if len(found) == 0:
                print('Please choose one of the following:')
                print(sd.query_devices())
                raise RuntimeError('Could not find a proper device')

            return found[0]
        else:
            return (sd.query_devices(deviceno,kind='input'),deviceno)
    
    def start(self):
        """
        
        """
        
        if self._running:
            raise RuntimeError('Stream already started')
        
        assert self._audiothread == None
        assert self._videothread == None
        
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
        stream = sd.InputStream(
            device=self.audiodeviceno,
            dtype=self.dtype,
            blocksize=blocksize,
            channels=self.nchannels,
            samplerate=self.samplerate,
            callback=self._audioCallback)
        
        with stream:
            with self._running_cond:
                while self._running:
                    self._running_cond.wait()
        print('stopped audiothread')

    def _videoThread(self):
        cap = cv.VideoCapture(self._video)
        if not cap.isOpened():
            cap.open()
        vframectr = 0
        loopctr = 0
        while self._running:
            ret, frame = cap.read()
            # print(frame.shape)
            if ret==True:
                if vframectr == 0:
                    self._video_started <<= True
                with self._callbacklock:
                    for cb in self._callbacks:
                        cb(AvType.video,frame,self._aframectr(),vframectr)
                vframectr += 1
                self._vframectr += 1
            else:
                
                if loopctr == 10:
                    print('Error: no video capture!')
            time.sleep(0.2)
            loopctr +=1

        cap.release()
        print('stopped videothread')

    def _audioCallback(self, indata, nframes, time, status):
        """This is called (from a separate thread) for each audio block."""
        if not self._video_started:
            return
        
        with self._callbacklock:
            for cb in self._callbacks:
                cb(AvType.audio,indata,self._aframectr(),self._vframectr())
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
