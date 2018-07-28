#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description:

Data Acquistiion (DAQ) device descriptors, and the DAQ devices themselves

"""
__all__ = ['DAQConfiguration', 'roga_plugndaq', 'default_soundcard']


class DAQConfiguration:
    def __init__(self, name,
                 cardname,
                 cardlongnamematch,
                 device_name,
                 en_format,
                 en_input_rate,
                 en_input_channels,

                 input_sensitivity,
                 input_gain_settings,
                 en_input_gain_setting,

                 en_output_rate,
                 en_output_channels):
        """
        Initialize a device descriptor

        Args:
            name: Name of the device to appear to the user
            cardname: ALSA name identifier
            cardlongnamematch: Long name according to ALSA
            device_name: ASCII name with which to open the device when connected
            en_format: index in the list of sample formats
            en_input_rate: index of enabled input sampling frequency [Hz]
                                in the list of frequencies.
            en_input_channels: list of channel indices which are used to
                                    acquire data from.
            input_sensitivity: List of sensitivity values, in units of [Pa^-1]
            input_gain_setting: If a DAQ supports it, list of indices which
                                corresponds to a position in the possible input
                                gains for each channel. Should only be not equal
                                to None when the hardware supports changing the
                                input gain.
            en_output_rate: index in the list of possible output sampling
                                 frequencies.
            en_output_channels: list of enabled output channels


        """
        self.name = name
        self.cardlongnamematch = cardlongnamematch
        self.cardname = cardname
        self.device_name = device_name
        self.en_format = en_format

        self.en_input_rate = en_input_rate
        self.en_input_channels = en_input_channels

        self.input_sensitivity = input_sensitivity
        self.input_gain_settings = input_gain_settings

        self.en_output_rate = en_output_rate
        self.en_output_channels = en_output_channels

    def __repr__(self):
        """
        String representation of configuration
        """
        rep = f"""Name: {self.name}
        Enabled input channels: {self.en_input_channels}
        Enabled input sampling frequency: {self.en_input_rate}
        Input gain settings: {self.input_gain_settings}
        Sensitivity: {self.input_sensitivity}
        """
        return rep


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


roga_plugndaq = DAQConfiguration(name='Roga-instruments Plug.n.DAQ USB',
                                 cardname='USB Audio CODEC',
                                 cardlongnamematch='Burr-Brown from TI USB'
                                 ' Audio CODEC',
                                 device_name='iec958:CARD=CODEC,DEV=0',
                                 en_format=0,
                                 en_input_rate=2,
                                 en_input_channels=[0],
                                 input_sensitivity=[1., 1.],
                                 input_gain_settings=[-20, 0, 20],
                                 en_input_gain_setting=[1, 1],
                                 en_output_rate=1,
                                 en_output_channels=[False, False]
                                 )

default_soundcard = DAQConfiguration(name="Default device",
                                     cardname=None,
                                     cardlongnamematch=None,
                                     device_name='default',
                                     en_format=0,
                                     en_input_rate=2,
                                     en_input_channels=[0, 1],
                                     input_sensitivity=[1., 1.],
                                     input_gain_settings=[0],
                                     en_input_gain_setting=[0, 0],
                                     en_output_rate=1,
                                     en_output_channels=[]
                                     )
configs = (roga_plugndaq, default_soundcard)
