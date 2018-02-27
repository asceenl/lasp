#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description: Designs FIR octave band FIR filters from 16Hz to 16 kHz for a
sampling frequency of 48 kHz.
"""
import numpy as np
from octave_filter_limits import octave_band_limits, G, fr
from matplotlib.pyplot import figure, close
from fir_design import freqResponse, bandpass_fir_design

L = 256 # Filter order
fs = 48000. # Sampling frequency

showfig = False
#showfig = True
maxdec = 3
close('all')

filters = {}
decimation = {}
b = 1
# Text corresponding to the nominal frequency
nominals = {4:'16k',
            3:'8k',
            2:'4k',
            1:'2k',
            0:'1k',
            -1:'500',
            -2:'250',
            -3:'125',
            -4:'63',
            -5:'31.5',
            -6:'16'}

# Factor with which to multiply the cut-on frequency of the FIR filter
cut_fac_l = {
4:0.995,
3:0.99,
2:0.98,
1:0.99,
0:0.98,
-1:0.96,
-2:.99,
-3:0.98,
-4:0.96,
-5:0.98,
-6:0.96
}

# Factor with which to multiply the cut-off frequency of the FIR filter
cut_fac_u = {
4:1.004,
3:1.006,
2:1.01,
1:1.006,
0:1.01,
-1:1.02,
-2:1.006,
-3:1.01,
-4:1.02,
-5:1.01,
-6:1.02
}

# Required decimation for each filter
decimation = {
4:[1],
3:[1],
2:[1],
1:[4],
0:[4],
-1:[4],
-2:[4,8],
-3:[4,8],
-4:[4,8],
-5:[4,8,4],
-6:[4,8,4]
        }

# Generate the header file
with open('../c/lasp_octave_fir.h','w') as hfile:
    hfile.write("""// lasp_octave_fir.h
//
// Author: J.A. de Jong - ASCEE
//
// Description: This is file is automatically generated from the
// octave_filter_design.py Python file.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_OCTAVE_FIR_H
#define LASP_OCTAVE_FIR_H
#include "lasp_types.h"

#define MAXDEC (%i) /// Size of the decimation factor array
#define OCTAVE_FIR_LEN (%i) /// Filter length

typedef struct {
    const char* nominal; /// Pointer to the nominal frequency text
    int x; /// 1000*G^x is the exact midband frequency, where G = 10^(3/10)
    int decimation_fac[MAXDEC]; // Array with decimation factors that need to be
                                // applied prior to filtering
    d h[OCTAVE_FIR_LEN];
} OctaveFIR;

extern __thread OctaveFIR OctaveFIRs[%i];
        
#endif // LASP_OCTAVE_FIR_H
//////////////////////////////////////////////////////////////////////
""" %(maxdec,L,len(decimation)))
    
# Generate the source code file
with open('../c/lasp_octave_fir.c','w') as cfile:
    cfile.write("""// octave_fir.c
//
// Author: J.A. de Jong - ASCEE
// 
// Description:
// This is an automatically generated file containing filter coefficients
// for octave filter banks.
//////////////////////////////////////////////////////////////////////
#include "lasp_octave_fir.h"

__thread OctaveFIR OctaveFIRs[%i] = {
""" %(len(decimation)))
    struct = ''
    for x in range(4,-7,-1):
        
        if showfig:
            fig = figure()
            ax = fig.add_subplot(111)
        
        dec = decimation[x]
        d = np.prod(dec)
        
        struct += '\n    {"%s",%i, {' %(nominals[x],x)
        for i in range(maxdec):
            try:
                struct += "%i," % dec[i]
            except:
                struct += "0,"
        struct = struct[:-1] # Strip off last,
        struct += '},'
        
        fd = fs/d
        
        # Exact midband frequency
        fm = G**(x)*fr
        
        # Cut-on frequency of the filter
        f1 = fm*G**(-1/(2*b))*cut_fac_l[x]
        # Cut-off frequency of the filter
        f2 = fm*G**(1/(2*b))*cut_fac_u[x]

        fir = bandpass_fir_design(L,fd,f1,f2)        
        
        struct += "{ "
        for i in fir:
            struct += "%0.16e," %i
        struct = struct[:-1] + " }},"
        
        freq = np.logspace(np.log10(f1/3),np.log10(f2*3),1000)
        H = freqResponse(fir,freq,fd)
        if showfig:        
            ax.semilogx(freq,20*np.log10(np.abs(H)))
            
            freq,ulim,llim = octave_band_limits(x)
            
            ax.semilogx(freq,ulim)
            ax.semilogx(freq,llim)
            ax.set_title('x = %i, fnom = %s' %(x,nominals[x]) )
        
            ax.set_ylim(-10,1)
            ax.set_xlim(f1/1.1,f2*1.1)
    
    struct+="\n};"
    cfile.write(struct)    
    
