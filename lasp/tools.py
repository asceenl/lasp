#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description:
"""
import numpy as np

def getFreq(fs, nfft):
    df = fs/nfft  # frequency resolution 
    K = nfft//2+1 # number of frequency bins
    return np.linspace(0, (K-1)*df, K)
