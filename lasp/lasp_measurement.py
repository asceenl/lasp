#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description: Measurement class
"""
import h5py as h5
import numpy as np
from .lasp_config import LASP_NUMPY_FLOAT_TYPE
import wave,os

class BlockIter:
    def __init__(self,nblocks,faudio):
        self.i = 0
        self.nblocks = nblocks
        self.fa = faudio
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == self.nblocks:
            raise StopIteration
        self.i+=1
        return self.fa[self.i-1][:,:]


def getSampWidth(dtype):
    if dtype == np.int32:
        return 4
    elif dtype == np.int16:
        return 2
    else:
        raise ValueError('Invalid data type: %s' %dtype)


def exportAsWave(fn,fs,data,force=False):
    if not '.wav' in fn[-4:]:
        fn += '.wav'

    nchannels = data.shape[1]
    sampwidth = getSampWidth(data.dtype)

    if os.path.exists(fn) and not force:
        raise RuntimeError('File already exists: %s', fn)

    with wave.open(fn,'w') as wf:
        wf.setparams((nchannels,sampwidth,fs,0,'NONE','NONE'))
        wf.writeframes(np.asfortranarray(data).tobytes())


class Measurement:
    def __init__(self, fn):

        if not '.h5' in fn:
            fn += '.h5'

        self.fn = fn

        f = h5.File(fn,'r+')
        self.f = f
        try:
            f['video']
            self.has_video = True
        except KeyError:
            self.has_video = False

        self.nblocks, self.blocksize, self.nchannels = f['audio'].shape
        dtype = f['audio'].dtype
        self.sampwidth = getSampWidth(dtype)

        self.samplerate = f.attrs['samplerate']
        self.N = (self.nblocks*self.blocksize)
        self.T = self.N/self.samplerate

        # comment = read-write thing
        try:
            self.f.attrs['comment']
        except KeyError:
            self.f.attrs['comment'] = ''

    @property
    def comment(self):
        return self.f.attrs['comment']

    @comment.setter
    def comment(self, cmt):
        self.f.attrs['comment'] = cmt

    @property
    def recTime(self):
        return (self.blocksize*self.nblocks)/self.samplerate

    @property
    def time(self):
        return self.f.attrs['time']

    @property
    def prms(self):
        try:
            return self._prms
        except AttributeError:
            pass
        sens = self.sensitivity
        pms = 0.
        for block in self.iterBlocks():
            pms += np.sum(block/sens)**2/self.N
        self._prms = np.sqrt(pms)
        return self._prms


    def praw(self,block=None):
        if block is not None:
            blocks = self.f['audio'][block]
        else:
            blocks = []
            for block in self.iterBlocks():
                blocks.append(block)
            blocks = np.asarray(blocks)

            blocks = blocks.reshape(self.nblocks*self.blocksize,
                                  self.nchannels)
        if blocks.dtype == np.int32:
            fac = 2**31
        elif blocks.dtype == np.int16:
            fac = 2**15
        else:
            raise RuntimeError('Unimplemented data type from recording: %s' %str(blocks.dtype))

        blocks = blocks.astype(LASP_NUMPY_FLOAT_TYPE)/fac/self.sensitivity

        return blocks

    def iterBlocks(self):
        return BlockIter(self.nblocks,self.f['audio'])

    @property
    def sensitivity(self):
        try:
            return self.f.attrs['sensitivity']
        except:
            return 1.0

    @sensitivity.setter
    def sensitivity(self, sens):
        self.f.attrs['sensitivity'] = sens

    def exportAsWave(self, fn=None, force=False):
        """
        Export measurement file as wave
        """
        if fn is None:
            fn = self.fn
            fn = os.path.splitext(fn)[0]

        if not '.wav' in fn[-4:]:
            fn += '.wav'

        if os.path.exists(fn) and not force:
            raise RuntimeError('File already exists: %s', fn)

        with wave.open(fn,'w') as wf:
            wf.setparams((self.nchannels,self.sampwidth,self.samplerate,0,'NONE','NONE'))
            for block in self.iterBlocks():
                wf.writeframes(np.asfortranarray(block).tobytes())


    def __del__(self):
        try:
            self.f.close()
        except AttributeError:
            pass
