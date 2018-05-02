#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description: LASP configuration
"""
import numpy as np

LASP_NUMPY_FLOAT_TYPE = np.float64

def zeros(shape):
    return np.zeros(shape, dtype=LASP_NUMPY_FLOAT_TYPE)
def ones(shape):
    return np.ones(shape, dtype=LASP_NUMPY_FLOAT_TYPE)
def empty(shape):
    return np.empty(shape, dtype=LASP_NUMPY_FLOAT_TYPE)

class config:
    default_figsize = (12,7)
    default_comment_lenght_measoverview = 50
    default_xlim_freq_plot = (50, 10000)
    default_ylim_freq_plot = (20, 100)
    versiontxt = '0.1'
