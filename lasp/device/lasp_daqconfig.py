#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description:

Data Acquistiion (DAQ) device descriptors, and the DAQ devices themselves

"""
from dataclasses import dataclass, field

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


    """
    input_device_name: bytes
    output_device_name: bytes
    en_bit_depth: int
    en_input_rate: int
    en_input_channels: list
    en_input_gain_settings: list = field(default_factory=list)
    en_output_rate: int = -1

    def match(self, device):
        """
        Returns True when a device specification dictionary matches to the
        configuration.

        Args:
            device: dictionary specifying device settings
        """
        match = True
        if self.cardlongnamematch is not None:
            match &= self.cardlongnamematch in device.cardlongname
        if self.cardname is not None:
            match &= self.cardname == device.cardname
        if self.device_name is not None:
            match &= self.device_name == device.device_name
        match &= self.en_format < len(device.available_formats)
        match &= self.en_input_rate < len(device.available_input_rates)
        match &= max(
            self.en_input_channels) < device.max_input_channels
        if len(self.en_output_channels) > 0:
            match &= max(
                self.en_output_channels) < device.max_output_channels
        match &= self.en_output_rate < len(
            device.available_output_rates)

        return match

    @staticmethod
    def emptyFromDeviceAndSettings(device):
        return DAQConfiguration(
                name = 'UNKNOWN'
                input_device_name = 
        

def findDAQDevice(config: DAQConfiguration) -> DeviceInfo:
    """
    Search for a DaQ device for the given configuration.

    Args:
        config: configuration to search a device for
    """
    devices = query_devices()

    for device in devices:
        if config.match(device):
            return device
    return None
