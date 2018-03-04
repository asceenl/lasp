#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 19:45:33 2018

@author: anne
"""
import numpy as np
from lasp import FilterBank, cls
cls()

nfft=50
P = 4
L = nfft-P+2
nfilters = 1
print('nfft:',nfft)

h = np.zeros((P,nfilters),order='F')
h[P-1,0] = 1.
#h[0,1] = 1.

fb = FilterBank(h,nfft)
# %%
#data = np.zeros(nfft//2,order = 'F')
data = np.zeros(L,order = 'F')
data[:] = 1
#data[:] = 1.
#data[1] = 1.
#data[2] = 1.

res = fb.filter_(data)
print('res shape %i:\n' %res.shape[0])
print(res)
#for i in range(4):
#    res = fb.filter_(data)
#    print('res:\n',res)