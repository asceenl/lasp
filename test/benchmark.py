#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 19:45:33 2018

@author: anne
"""
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

import numpy as np
from beamforming import Fft

nfft=2**17

print('nfft:',nfft)
nchannels = 50
number_run = 10

t = np.linspace(0,1,nfft+1)

# Using transpose to get the strides right
x = np.random.randn(nchannels,nfft).T


import time
start = time.time()
for i in range(number_run):
    X = np.fft.rfft(x,axis=0)
end = time.time()
print("Time numpy fft:",end-start)
# X =np.fft.fft(x)
#X =np.fft.rfft(x)

fft = Fft(nfft,nchannels)

start = time.time()
for i in range(number_run):
    # print('--- run %i' %i)
    fft.fft(x)
end = time.time()

print("Time ASCEE fft:",end-start)


