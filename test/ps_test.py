#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 19:45:33 2018

@author: anne
"""
import numpy as np
from beamforming import Fft, PowerSpectra, cls
cls
nfft=2048
print('nfft:',nfft)
#print(nfft)
nchannels = 2

t = np.linspace(0,1,nfft+1)
# print(t)
x1 = (np.cos(4*np.pi*t[:-1])+3.2*np.sin(6*np.pi*t[:-1]))[:,np.newaxis]+10
x = np.vstack([x1.T]*nchannels).T
# Using transpose to get the strides right
x = np.random.randn(nchannels,nfft).T
print("strides: ",x.strides)
# x.strides = (8,nfft*8)x
# print("signal:",x)

xms = np.sum(x**2,axis=0)/nfft
print('Total signal power time domain: ', xms)

X = np.fft.rfft(x,axis=0)
# X =np.fft.fft(x)
#X =np.fft.rfft(x)

# print(X)
Xs = 2*X/nfft
Xs[np.where(np.abs(Xs) < 1e-10)] = 0
Xs[0] /= np.sqrt(2)
Xs[-1] /= np.sqrt(2)
# print('single sided amplitude spectrum:\n',Xs)


power = Xs*np.conj(Xs)/2

# print('Frequency domain signal power\n', power)
print('Total signal power', np.sum(power,axis=0).real)

pstest = PowerSpectra(nfft,nchannels)
ps = pstest.compute(x)

fft = Fft(nfft,nchannels)
fft.fft(x)

ps[np.where(np.abs(ps) < 1e-10)] = 0+0j

print('our ps: \n' , ps)

print('Our total signal power: ',np.sum(ps,axis=0).real)
