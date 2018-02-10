#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 19:45:33 2018

@author: anne
"""
import numpy as np
from beamforming import Fft

nfft=6
print('nfft:',nfft)
print(nfft)
nchannels = 1

t = np.linspace(0,1,nfft+1)[:-1]
# print(t)
x1 = 1+np.sin(2*np.pi*t)+3.2*np.cos(2*np.pi*t)+np.sin(7*np.pi*t)
x = np.vstack([x1.T]*nchannels).T
# Using transpose to get the strides right
x = np.random.randn(nchannels,nfft).T
# x.strides = (8,nfft*8)x
# print("signal:",x)

X = np.fft.rfft(x,axis=0)
print('Numpy fft')
print(X)

fft = Fft(nfft)
Y = fft.fft(x)
print('Fftpack fft')
print(Y)
print('end python script')
