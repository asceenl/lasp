#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description: Measurement class

The ASCEE hdf5 measurement file format contains the following fields:

- Attributes:

'samplerate': The audio data sample rate in Hz.
'nchannels': The number of audio channels in the file
'sensitivity': (Optionally) the stored sensitivity of the record channels.
               This can be a single value, or an array of sensitivities for
               each channel. Both representations are allowed.

- Datasets:

'audio': 3-dimensional array of blocks of audio data. The first axis is  the
block index, the second axis the sample number and the third axis is the channel
number. The data type is either int16, int32 or float64 / float32. In case the
data is stored as integers. The raw data should be scaled with the maximum value
that can be stored for the integer bit depth to get a number between -1.0 and
1.0.

'video': 4-dimensional array of video frames. The first index is the frame
         number, the second the x-value of the pixel and the third is the
         y-value of the pixel. Then, the last axis is the color. This axis has
         length 3 and the colors are stored as (r,g,b). Where typically a
         color depth of 256 is used (np.uint8 data format)

The video dataset can possibly be not present in the data.



"""
__all__ = ['Measurement']
from contextlib import contextmanager
import h5py as h5
import numpy as np
from .lasp_config import LASP_NUMPY_FLOAT_TYPE
import wave
import os


class BlockIter:
    """
    Iterate over the blocks in the audio data of a h5 file
    """

    def __init__(self, f):
        """
        Initialize a BlockIter object

        Args:
            faudio: Audio dataset in the h5 file, accessed as f['audio']
        """
        self.i = 0
        self.nblocks = f['audio'].shape[0]
        self.fa = f['audio']

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return the next block
        """
        if self.i == self.nblocks:
            raise StopIteration
        self.i += 1
        return self.fa[self.i-1][:, :]


def getSampWidth(dtype):
    """
    Returns the width of a single sample in bytes.

    Args:
        dtype: numpy dtype

    Returns:
        Size of a sample in bytes (int)
    """
    if dtype == np.int32:
        return 4
    elif dtype == np.int16:
        return 2
    elif dtype == np.float64:
        return 8
    else:
        raise ValueError('Invalid data type: %s' % dtype)


def exportAsWave(fn, fs, data, force=False):
    if not '.wav' in fn[-4:]:
        fn += '.wav'

    nchannels = data.shape[1]
    sampwidth = getSampWidth(data.dtype)

    if os.path.exists(fn) and not force:
        raise RuntimeError('File already exists: %s', fn)

    with wave.open(fn, 'w') as wf:
        wf.setparams((nchannels, sampwidth, fs, 0, 'NONE', 'NONE'))
        wf.writeframes(np.asfortranarray(data).tobytes())


class Measurement:
    """
    Provides access to measurement data stored in the h5 measurement file
    format.
    """

    def __init__(self, fn):
        """
        Initialize a Measurement object based on the filename
        """
        if '.h5' not in fn:
            fn += '.h5'

        # Full filepath
        self.fn = fn

        # Base filename
        self.fn_base = os.path.split(fn)[1]

        # Open the h5 file in read-plus mode, to allow for changing the
        # measurement comment.
        with h5.File(fn, 'r+') as f:
            # Check for video data
            try:
                f['video']
                self.has_video = True
            except KeyError:
                self.has_video = False

            self.nblocks, self.blocksize, self.nchannels = f['audio'].shape
            dtype = f['audio'].dtype
            self.sampwidth = getSampWidth(dtype)

            self.samplerate = f.attrs['samplerate']
            self.N = (self.nblocks*self.blocksize)
            self.T = self.N/self.samplerate

            # comment = read-write thing
            try:
                self._comment = f.attrs['comment']
            except KeyError:
                f.attrs['comment'] = ''
                self._comment = ''

            # Sensitivity
            try:
                self._sens = f.attrs['sensitivity']
            except KeyError:
                self._sens = np.ones(self.nchannels)

            self._time = f.attrs['time']

    @contextmanager
    def file(self, mode='r'):
        """
        Contextmanager which opens the storage file and yields the file.

        Args:
            mode: Opening mode for the file. Should either be 'r', or 'r+'
        """
        if mode not in ('r','r+'):
            raise ValueError('Invalid file opening mode.')
        with h5.File(self.fn, mode) as f:
            yield f

    @property
    def comment(self):
        return self._comment

    @comment.setter
    def comment(self, cmt):
        with self.file('r+') as f:
            f.attrs['comment'] = cmt
            self._comment = cmt

    @property
    def recTime(self):
        """
        Returns the total recording time of the measurement, in float seconds.
        """
        return self.blocksize*self.nblocks/self.samplerate

    @property
    def time(self):
        """
        Returns the measurement time in seconds since the epoch.
        """
        return self._time

    @property
    def prms(self):
        """
        Returns the root mean square of the uncalibrated rms sound pressure
        level (equivalend SPL).

        Returns:
            1D array with rms values for each channel
        """
        #
        try:
            return self._prms
        except AttributeError:
            pass

        sens = self._sens
        pms = 0.
        for block in self.iterBlocks():
            pms += np.sum(block/sens[np.newaxis, :], axis=0)**2/self.N
        self._prms = np.sqrt(pms)
        return self._prms

    def praw(self, block=None):
        """
        Returns the raw uncalibrated data, converted to floating point format.
        """
        if block is not None:
            with self.file() as f:
                blocks = f['audio'][block]
        else:
            blocks = []
            with self.file() as f:
                for block in self.iterBlocks(f):
                    blocks.append(block)
            blocks = np.asarray(blocks)
            blocks = blocks.reshape(self.nblocks*self.blocksize,
                                    self.nchannels)

        # When the data is stored as integers, we assume dB full-scale scaling.
        # Hence, when we convert the data to floats, we divide by the maximum
        # possible value.
        if blocks.dtype == np.int32:
            fac = 2**31
        elif blocks.dtype == np.int16:
            fac = 2**15
        elif blocks.dtype == np.float64:
            fac = 1.0
        else:
            raise RuntimeError(
                f'Unimplemented data type from recording: {blocks.dtype}.')
        sens = self._sens
        blocks = blocks.astype(LASP_NUMPY_FLOAT_TYPE)/fac/sens[np.newaxis, :]

        return blocks

    def iterBlocks(self, opened_file):
        """
        Iterate over all the audio blocks in the opened file

        Args:
            opened_file: The h5File with the data
        """
        return BlockIter(opened_file)

    @property
    def sensitivity(self):
        """
        Sensitivity of the data in Pa^-1, from floating point data scaled
        between -1.0 and 1.0 to Pascal. If the sensitivity is not stored in
        the measurement file, this function returns 1.0
        """
        return self._sens

    @sensitivity.setter
    def sensitivity(self, sens):
        """
        Set the sensitivity of the measurement in the file

        Args:
            sens: sensitivity data, should be a float, or an array of floats
                  equal to the number of channels.
        """
        if isinstance(sens, float):
            sens = sens*np.ones(self.nchannels)

        valid = sens.ndim == 1
        valid &= sens.shape[0] == self.nchannels
        valid &= isinstance(sens.dtype, float)
        if not valid:
            raise ValueError('Invalid sensitivity value(s) given')
        with self.file('r+') as f:
            f.attrs['sensitivity'] = sens

    def exportAsWave(self, fn=None, force=False, sampwidth=None):
        """
        Export measurement file as wave. In case the measurement data is stored
        as floats, the values are scaled between 0 and 1

        Args:
            fn: If given, this will be the filename to write to. If the
            filename does not end with '.wav', this extension is added.

            force: If True, overwrites any existing files with the given name
            , otherwise a RuntimeError is raised.

            sampwidth: sample width in bytes with which to export the data.
            This should only be given in case the measurement data is stored as
            floating point values, otherwise an

        """
        if fn is None:
            fn = self.fn
            fn = os.path.splitext(fn)[0]

        if '.wav' not in fn[-4:]:
            fn += '.wav'

        if os.path.exists(fn) and not force:
            raise RuntimeError(f'File already exists: {fn}')
        with self.file() as f:
            audio = self.f['audio'][:]

        if isinstance(audio.dtype, float):
            if sampwidth is None:
                raise ValueError('sampwidth parameter should be given '
                                 'for float data in measurement file')
            elif sampwidth == 2:
                itype = np.int16
            elif sampwidth == 4:
                itype = np.int32
            else:
                raise ValueError('Invalid sample width, should be 2 or 4')

            # Find maximum
            max = 0.
            for block in self.iterBlocks():
                blockmax = np.max(np.abs(block))
                if blockmax > max:
                    max = blockmax
            # Scale with maximum value only if we have a nonzero maximum value.
            if max == 0.:
                max = 1.
            scalefac = 2**(8*sampwidth)/max

        with wave.open(fn, 'w') as wf:
            wf.setparams((self.nchannels, self.sampwidth,
                          self.samplerate, 0, 'NONE', 'NONE'))
            for block in self.iterBlocks():
                if isinstance(block.dtype, float):
                    # Convert block to integral data type
                    block = (block*scalefac).astype(itype)
                wf.writeframes(np.asfortranarray(block).tobytes())

    @staticmethod
    def fromtxt(fn, skiprows, samplerate, sensitivity, mfn=None,
                timestamp=None,
                delimiter='\t', firstcoltime=True):
        """
        Converts a txt file to a LASP Measurement object and returns the
        measurement.

        Args:
            fn: Filename of text file
            skiprows: Number of header rows in text file to skip
            samplerate: Sampling frequency in [Hz]
            sensitivity: 1D array of channel sensitivities
            mfn: Filepath where measurement file is stored. If not given,
            a h5 file will be created along fn, which shares its basename
            timestamp: If given, a custom timestamp for the measurement
            (integer containing seconds since epoch). If not given, the
            timestamp is obtained from the last modification time.
            delimiter: Column delimiter
            firstcoltime: If true, the first column is the treated as the
            sample time.

        """
        if not os.path.exists(fn):
            raise ValueError(f'File {fn} does not exist.')
        if timestamp is None:
            timestamp = os.path.getmtime(fn)
        if mfn is None:
            mfn = os.path.splitext(fn)[0] + '.h5'

        dat = np.loadtxt(fn, skiprows=skiprows, delimiter=delimiter)
        if firstcoltime:
            time = dat[:, 0]
            if not np.isclose(time[1] - time[0], 1/samplerate):
                raise ValueError('Samplerate given does not agree with '
                                 'samplerate in file')
            dat = dat[:, 1:]
        nchannels = dat.shape[1]

        with h5.File(mfn, 'w') as hf:
            hf.attrs['samplerate'] = samplerate
            hf.attrs['sensitivity'] = sensitivity
            hf.attrs['time'] = timestamp
            hf.attrs['blocksize'] = 1
            hf.attrs['nchannels'] = nchannels
            ad = hf.create_dataset('audio',
                                   (1, dat.shape[0], dat.shape[1]),
                                   dtype=dat.dtype,
                                   maxshape=(1, dat.shape[0], dat.shape[1]),
                                   compression='gzip')
            ad[0] = dat
        return Measurement(mfn)

    # def __del__(self):
    #     self.f.close()
