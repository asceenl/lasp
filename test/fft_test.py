#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 19:45:33 2018

@author: anne
"""
import numpy as np
from lasp import Fft

nfft=9
print('nfft:',nfft)
print(nfft)
nchannels = 4

t = np.linspace(0,1,nfft+1)[:-1]
# print(t)
#x1 = 1+np.sin(2*np.pi*t)+3.2*np.cos(2*np.pi*t)+np.sin(7*np.pi*t)
#x1 = np.sin(2*np.pi*t)
x1 = 1+0*t
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
print('Beamforming fft')
print(Y)

x2 = fft.ifft(Y)
print('normdiff:',np.linalg.norm(x2-x))
print('end python script')
