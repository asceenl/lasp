#!/usr/bin/python
"""
Bin narrow band power in octave/third octave band data
"""
from lasp.filter.bandpass_fir import (OctaveBankDesigner, 
                                      ThirdOctaveBankDesigner)
import numpy as np
import warnings


def binPower(freq, narrow_power, band=3, start_band=-16, stop_band=13):
    """
    Apply binning to narrow band frequency domain power results
    
    Args:
        freq: Array of frequency indices
        narrow_power: narrow-band power values in units of [W] or [Pa^2]
        band: 1, or 3
    
    Returns:
        ( ['25', '31.5', '40', '50', ... ],
          [float(power_25), float(power_31p5), ...]) putting NAN values where
           inaccurate.
    """

    if band == 3:
        designer = ThirdOctaveBankDesigner()
    elif band == 1:
        designer = OctaveBankDesigner()
    else:
        raise ValueError("Parameter 'Band' should be either '1', or '3'")

    freq = np.copy(freq)
    narrow_power = np.copy(narrow_power)

    
    # Exact midband, lower and upper frequency limit of each band
    fm = [designer.fm(x) for x in range(start_band, stop_band+1)]
    fl = [designer.fl(x) for x in range(start_band, stop_band+1)]
    fu = [designer.fu(x) for x in range(start_band, stop_band+1)]
    fex = [designer.nominal_txt(x) for x in range(start_band, stop_band+1)]
#    print(fl)
    binned_power = np.zeros(len(fm), dtype=float)
    ## Start: linear interpolation between bins while Parseval is conserved

    # current frequency resolution
    df_old = freq[1]-freq[0]
    
    # preferred new frequency resolution
    df_new = .1

    # ratio of resolutions

    ratio = int(df_old/df_new)
    # calculate new frequency bins
    freq_new = np.linspace(freq[0],freq[-1],(len(freq)-1)*ratio+1)         
    
    # calculate the new bin data
    interp_power = np.interp(freq_new, freq, narrow_power)/ratio
    
    # adapt first and last bin values so that Parseval still holds
    interp_power[0] = binned_power[0]*(1.+1./ratio)/2.
    interp_power[-1] = binned_power[-1]*(1.+1./ratio)/2.    

    # check if Parseval still holds        
    # print(np.sum(y, axis=0))
    # print(np.sum(y_new, axis=0))
    
    ## Stop: linear interpolation between bins while Parseval is conserved        
    binned_power = np.zeros(len(fm), dtype=float)
    for k in range(len(fm)):
#        print(k)
        # find the bins which are in the corresponding band        
        bins = (fl[k] <= freq_new) & (freq_new < fu[k])
#        print(bins)
        # sum the output values of these bins to obtain the band value
        binned_power[k] = np.sum(interp_power[bins], axis=0)
        # if no frequency bin falls in a certain band, skip previous bands 
#        if not any(bins):
#            binned_power[0:k+1] = np.nan

    # check if data is valid
    if(np.isnan(binned_power).all()):
        warnings.warn('Invalid frequency array, we cannot bin these values')

    return fm, fex, binned_power