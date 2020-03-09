#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description:

Data Acquistiion (DAQ) device descriptors, and the DAQ devices themselves

"""
from dataclasses import dataclass, field
import numpy as np


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
        duplex_mode: Set device to duplex mode, if possible
        monitor_gen: If set to true, add monitor channel to recording.
        input_device_name: ASCII name with which to open the device when connected
        outut_device_name: ASCII name with which to open the device when connected

        ==============================
        en_format: index of the format in the list of sample formats
        en_input_rate: index of enabled input sampling frequency [Hz]
                            in the list of frequencies.
        input_channel_configs: list of channel indices which are used to
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

    input_channel_configs: list = None
    monitor_gen: bool = False


    def __post_init__(self):
        """
        We do a check here to see whether the list of enabled channels is
        contiguous. Non-contiguous is not yet implemented in RtAudio backend.
        """
        en_input = self.input_channel_configs
        first_ch_enabled_found = False
        ch_disabled_found_after = False
        print(en_input)
        for ch in en_input:
            if ch.channel_enabled:
                first_ch_enabled_found = True
                if ch_disabled_found_after:
                    raise ValueError('Error: non-contiguous array of channels'
                                    ' found. This is not yet implemented in'
                                     ' backend')
            else:
                if first_ch_enabled_found:
                    ch_disabled_found_after = True

    def firstEnabledInputChannelNumber(self):
        """
        Returns the channel number of the first enabled channel. Returns -1 if
        no channels are enabled.
        """
        for i, ch in enumerate(self.input_channel_configs):
            if ch.channel_enabled:
                return i
        return -1


    def getEnabledChannels(self):
        en_channels = []
        for chan in self.input_channel_configs:
            if chan.channel_enabled:
                en_channels.append(chan)
        return en_channels

    def getEnabledChannelSensitivities(self):
        return np.array(
                [float(channel.sensitivity) for channel in
                    self.getEnabledChannels()])

    @staticmethod
    def loadConfigs():
        """
        Returns a list of currently available configurations
        """
        with lasp_shelve() as sh:
            return sh.load('daqconfigs', {})

    def saveConfig(self, name):
        with lasp_shelve() as sh:
            cur_configs = self.loadConfigs()
            cur_configs[name] = self
            sh.store('daqconfigs', cur_configs)

    @staticmethod
    def deleteConfig(name):
        with lasp_shelve() as sh:
            cur_configs = DAQConfiguration.loadConfigs()
            del cur_configs[name]
            sh.store('daqconfigs', cur_configs)

