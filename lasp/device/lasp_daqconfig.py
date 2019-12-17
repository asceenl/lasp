#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description:

Data Acquistiion (DAQ) device descriptors, and the DAQ devices themselves

"""
from dataclasses import dataclass, field

@dataclass
class DeviceInfo:
    index: int
    probed: bool
    name: str
    outputchannels: int
    inputchannels: int
    duplexchannels: int
    samplerates: list
    sampleformats: list
    prefsamplerate: int


from ..lasp_common import lasp_shelve

@dataclass
class DAQInputChannel:
    channel_enabled: bool
    channel_name: str
    sensitivity: float




@dataclass
class DAQConfiguration:
    """
    Initialize a device descriptor

    Args:
        input_device_name: ASCII name with which to open the device when connected
        outut_device_name: ASCII name with which to open the device when connected

        ==============================
        en_format: index of the format in the list of sample formats
        en_input_rate: index of enabled input sampling frequency [Hz]
                            in the list of frequencies.
        en_input_channels: list of channel indices which are used to
                                acquire data from.
        input_sensitivity: List of sensitivity values, in units of [Pa^-1]
        input_gain_settings: If a DAQ supports it, list of indices which
                            corresponds to a position in the possible input
                            gains for each channel. Should only be not equal
                            to None when the hardware supports changing the
                            input gain.
        en_output_rate: index in the list of possible output sampling
                             frequencies.
        en_output_channels: list of enabled output channels
        ===============================


    """
    duplex_mode: bool

    input_device_name: str
    output_device_name: str

    en_input_sample_format: str
    en_output_sample_format: str
    en_input_rate: int
    en_output_rate: int

    en_input_channels: list

    @staticmethod
    def loadConfigs():
        """
        Returns a list of currently available configurations
        """
        with lasp_shelve() as sh:
            return sh['daqconfigs'] if 'daqconfigs' in sh.keys() else {}

    def saveConfig(self, name):
        with lasp_shelve() as sh:
            if 'daqconfigs' in sh.keys():
                cur_configs = sh['daqconfigs']
            else:
                cur_configs = {}
            cur_configs[name] = self
            sh['daqconfigs'] = cur_configs

    @staticmethod
    def deleteConfig(name):
        with lasp_shelve() as sh:
            cur_configs = sh['daqconfigs']
            del cur_configs[name]
            sh['daqconfigs'] = cur_configs

