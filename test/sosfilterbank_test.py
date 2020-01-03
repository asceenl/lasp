#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 19:45:33 2018

@author: anne
"""
import numpy as np
from lasp.wrappers import SosFilterBank

fb = SosFilterBank(1, 2)

newfilter1 = np.array([0,1.,0.,1.,0.,0.])[:,np.newaxis] # Unit sample delay
newfilter2 = np.array([0,1.,0.,1.,0.,0.])[:,np.newaxis] # Unit sample delay
newfilter = np.vstack((newfilter1, newfilter2))

fb.setFilter(0, newfilter)
x = np.zeros(10)
x[5]=1
x[8] = 3
x = x[:,np.newaxis]
y = fb.filter_(x)
print(x)
print(y)
y = fb.filter_(x)
print(y)
