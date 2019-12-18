#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:49:27 2018

@author: J.A. de Jong - ASCEE

Description: apply calibration to measurement files.

"""
import numpy as np
import argparse
from lasp.lasp_measurement import Measurement
from lasp.lasp_common import P_REF
import os

spl_default = 94.
gain_default = 0.

parser = argparse.ArgumentParser('Calibrate device using'
                                 ' calibration measurement')

parser.add_argument('--gain-setting', '-g',
                    help='DAQ Input gain setting during calibration in [dB]'
                    + f' default = {gain_default} dB.',
                    type=float, default=gain_default)

parser.add_argument(
    'fn', help='File name of calibration measurement', type=str, default=None)

parser.add_argument('--channel', help='Channel of the device to calibrate, '
                    + 'default = 0',
                    type=int, default=0)

parser.add_argument('--spl', '-s', help='Applied sound pressure level to the'
                    f' microphone in dB, default = {spl_default}',
                    default=spl_default)
args = parser.parse_args()

m = Measurement(args.fn)
nchannels = m.nchannels

# Reset measurement sensitivity, in case it was set wrongly
m.sensitivity = np.ones(nchannels)

# Compute Vrms
Vrms = m.prms * 10**(args.gain_setting/20.)

prms = P_REF*10**(args.spl/20)

sens = Vrms / prms

print(f'Computed sensitivity: {sens[args.channel]:.5} V/Pa')
if args.fn:
    print('Searching for files in directory to apply sensitivity value to...')
    dir_ = os.path.dirname(args.fn)
    for f in os.listdir(dir_):
        yn = input(f'Apply sensitivity to {f}? [Y/n]')
        if yn in ['', 'Y', 'y']:
            meas = Measurement(os.path.join(dir_, f))
            meas.sensitivity = sens
