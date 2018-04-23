#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description: Limit lines for class 1 octave band filter limits according to
the ICS 17.140.50 standard.
"""
__all__ = ['G','fr','octave_band_limits']
import numpy as np

# Reference frequency
fr = 1000.
G = 10**(3/10)

def octave_band_limits(x):
    
    # Exact midband frequency
    fm = G**(x)*fr

    G_power_values_pos = [0,1/8,1/4,3/8,1/2,1/2,1,2,3,4]
    G_power_values_neg = [-i for i in G_power_values_pos]
    G_power_values_neg.reverse()
    G_power_values = G_power_values_neg[:-1] + G_power_values_pos
    
    mininf = -1e300
    
    lower_limits_pos = [-0.3,-0.4,-0.6,-1.3,-5.0,-5.0]+ 4*[mininf]
    lower_limits_neg =  lower_limits_pos[:]
    lower_limits_neg.reverse()
    lower_limits = np.asarray(lower_limits_neg[:-1] + lower_limits_pos)
    
    upper_limits_pos = [0.3]*5 + [-2,-17.5,-42,-61,-70]
    upper_limits_neg =  upper_limits_pos[:]
    upper_limits_neg.reverse()
    upper_limits = np.asarray(upper_limits_neg[:-1] + upper_limits_pos)
    
    freqs = fm*G**np.asarray(G_power_values)

    return freqs,upper_limits,lower_limits

if __name__ == '__main__':
    
    from asceefigs.plot import close, Figure
    close('all')
    freqs,upper_limits,lower_limits = octave_band_limits(0)

    f = Figure()
    f.semilogx(freqs,lower_limits)
    f.semilogx(freqs,upper_limits)
    
    f.ylim(-80,1)