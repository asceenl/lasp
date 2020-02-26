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
               This can be a single value, or a list of sensitivities for
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

__all__ = ['Measurement', 'scaleBlockSens']
from contextlib import contextmanager
import h5py as h5
import numpy as np
from .lasp_config import LASP_NUMPY_FLOAT_TYPE
import wave
import os
import time


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
        return self.fa[self.i - 1][:, :]


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


def scaleBlockSens(block, sens):
    """
    Scale a block of raw data to return raw acoustic
    pressure data.

    Args:
        block: block of raw data with integer data type
        sensitivity: array of sensitivity coeficients for
        each channel

    """
    assert sens.ndim == 1
    assert sens.size == block.shape[1]
    if np.issubdtype(block.dtype.type, np.integer):
        sw = getSampWidth(block.dtype)
        fac = 2**(8 * sw - 1) - 1
    else:
        fac = 1.
    return block.astype(LASP_NUMPY_FLOAT_TYPE) / fac / sens[np.newaxis, :]


def exportAsWave(fn, fs, data, force=False):
    if '.wav' not in fn[-4:]:
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
            self.N = (self.nblocks * self.blocksize)
            self.T = self.N / self.samplerate

            try:
                self._channel_names = f.attrs['channel_names']
            except KeyError:
                # No channel names found in measurement file
                self._channel_names = [f'Unnamed {i}' for i in range(self.nchannels)]

            # comment = read-write thing
            try:
                self._comment = f.attrs['comment']
            except KeyError:
                f.attrs['comment'] = ''
                self._comment = ''

            # Sensitivity
            try:
                sens = f.attrs['sensitivity']
                self._sens = sens * \
                    np.ones(self.nchannels) if isinstance(
                        sens, float) else sens
            except KeyError:
                self._sens = np.ones(self.nchannels)

            self._time = f.attrs['time']

    @property
    def name(self):
        """
        Returns filename base without extension
        """
        return os.path.splitext(self.fn_base)[0]

    @property
    def channelNames(self):
        return self._channel_names

    @contextmanager
    def file(self, mode='r'):
        """
        Contextmanager which opens the storage file and yields the file.

        Args:
            mode: Opening mode for the file. Should either be 'r', or 'r+'
        """
        if mode not in ('r', 'r+'):
            raise ValueError('Invalid file opening mode.')
        with h5.File(self.fn, mode) as f:
            yield f

    @property
    def comment(self):
        """
        Return the measurement comment

        Returns:
            The measurement comment (text string)
        """
        return self._comment

    @comment.setter
    def comment(self, cmt):
        """
        Set the measurement comment

        Args:
            cmt: Comment text string to set
        """
        with self.file('r+') as f:
            # Update comment attribute in the file
            f.attrs['comment'] = cmt
            self._comment = cmt

    @property
    def recTime(self):
        """
        Returns
            the total recording time of the measurement, in float seconds.
        """
        return self.blocksize * self.nblocks / self.samplerate

    @property
    def time(self):
        """
        Returns the measurement time in seconds since the epoch.
        """
        return self._time

    def scaleBlock(self, block):
        """
        When the data is stored as integers, we assume dB full-scale scaling.
        Hence, when we convert the data to floats, we divide by the maximum
        possible value.

        Returns:
            Block of measurement data, scaled using sensitivity values and
            retured as floating point values
        """
        return scaleBlockSens(block, self.sensitivity)

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

        pms = 0.

        with self.file() as f:
            for block in self.iterBlocks(f):
                block = self.scaleBlock(block)
                pms += np.sum(block**2, axis=0) / self.N
        self._prms = np.sqrt(pms)
        return self._prms

    def praw(self, block=None):
        """
        Returns the uncalibrated acoustic pressure signal, converted to
        floating  point acoustic pressure values [Pa].
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
            blocks = blocks.reshape(self.nblocks * self.blocksize,
                                    self.nchannels)

        # Apply scaling (sensitivity, integer -> float)
        blocks = self.scaleBlock(blocks)
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

    def checkOverflow(self):
        """
        Coarse check for overflow in measurement

        Return:
            True if overflow is possible, else False

        """

        with self.file() as f:
            for block in self.iterBlocks(f):
                dtype = block.dtype
                if dtype.kind == 'i':
                    # minvalue = np.iinfo(dtype).min
                    maxvalue = np.iinfo(dtype).max
                    if np.max(np.abs(block)) >= 0.9*maxvalue:
                        return True
                else:
                    # Cannot check for floating point values.
                    return False
            return False


    @sensitivity.setter
    def sensitivity(self, sens):
        """
        Set the sensitivity of the measurement in the file

        Args:
            sens: sensitivity data, should be a float, or an array of floats
                  equal to the number of channels.
        """
        if isinstance(sens, float):
            sens = sens * np.ones(self.nchannels)

        valid = sens.ndim == 1
        valid &= sens.shape[0] == self.nchannels
        valid &= sens.dtype == float
        if not valid:
            raise ValueError('Invalid sensitivity value(s) given')
        with self.file('r+') as f:
            f.attrs['sensitivity'] = sens
        self._sens = sens

    def exportAsWave(self, fn=None, force=False, newsampwidth=2, normalize=True):
        """
        Export measurement file as wave. In case the measurement data is stored
        as floats, the values are scaled between 0 and 1

        Args:
            fn: If given, this will be the filename to write to. If the
            filename does not end with '.wav', this extension is added.

            force: If True, overwrites any existing files with the given name
            , otherwise a RuntimeError is raised.

            newsampwidth: sample width in bytes with which to export the data.
            This should only be given in case the measurement data is stored as
            floating point values, otherwise an error is thrown

            normalize: If set: normalize the level to something sensible.


        """
        if fn is None:
            fn = self.fn
            fn = os.path.splitext(fn)[0]

        if '.wav' not in fn[-4:]:
            fn += '.wav'

        if os.path.exists(fn) and not force:
            raise RuntimeError(f'File already exists: {fn}')

        with self.file() as f:

            audio = f['audio']
            oldsampwidth = getSampWidth(audio.dtype)

            max_ = 1.
            if normalize:
                # Find maximum value
                for block in self.iterBlocks(f):
                    blockmax = np.max(np.abs(block))
                    max_ = blockmax if blockmax > max_ else max_
                # Scale with maximum value only if we have a nonzero maximum value.
                if max_ == 0.:
                    max_ = 1.

            if newsampwidth == 2:
                newtype = np.int16
            elif newsampwidth == 4:
                newtype = np.int32
            else:
                raise ValueError('Invalid sample width, should be 2 or 4')

            scalefac = 2**(8*(newsampwidth-oldsampwidth))
            if normalize or isinstance(audio.dtype, float):
                scalefac *= .01*max

            with wave.open(fn, 'w') as wf:
                wf.setparams((self.nchannels, self.sampwidth,
                              self.samplerate, 0, 'NONE', 'NONE'))
                for block in self.iterBlocks(f):
                    # Convert block to integral data type
                    block = (block*scalefac).astype(newtype)
                    wf.writeframes(np.asfortranarray(block).tobytes())

    @staticmethod
    def fromtxt(fn,
                skiprows,
                samplerate,
                sensitivity,
                mfn=None,
                timestamp=None,
                delimiter='\t',
                firstcoltime=True):
        """
        Converts a txt file to a LASP Measurement file, opens the associated
        Measurement object and returns it. The measurement file will have
        the same file name as the txt file, except with h5 extension.

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
        else:
            mfn = os.path.splitext(mfn)[0] + '.h5'

        dat = np.loadtxt(fn, skiprows=skiprows, delimiter=delimiter)
        if firstcoltime:
            time = dat[:, 0]
            if not np.isclose(time[1] - time[0], 1 / samplerate):
                raise ValueError('Samplerate given does not agree with '
                                 'samplerate in file')

            # Chop off first column
            dat = dat[:, 1:]
        nchannels = dat.shape[1]
        if nchannels != sensitivity.shape[0]:
            raise ValueError(
                f'Invalid sensitivity length given. Should be: {nchannels}')

        with h5.File(mfn, 'w') as hf:
            hf.attrs['samplerate'] = samplerate
            hf.attrs['sensitivity'] = sensitivity
            hf.attrs['time'] = timestamp
            hf.attrs['blocksize'] = 1
            hf.attrs['nchannels'] = nchannels
            ad = hf.create_dataset('audio', (1, dat.shape[0], dat.shape[1]),
                                   dtype=dat.dtype,
                                   maxshape=(1, dat.shape[0], dat.shape[1]),
                                   compression='gzip')
            ad[0] = dat
        return Measurement(mfn)
 

    @staticmethod
    def fromnpy(data,
                samplerate,
                sensitivity,
                mfn,
                timestamp=None):
        """
        Converts a numpy array to a LASP Measurement file, opens the associated
        Measurement object and returns it. The measurement file will have
        the same file name as the txt file, except with h5 extension.


        Args:
            data: Numpy array, first column is sample, second is channel. Can 
            also be specified with a single column for single-channel data
            samplerate: Sampling frequency in [Hz]
            sensitivity: 1D array of channel sensitivities [Pa^-1]
            mfn: Filepath where measurement file is stored.
            timestamp: If given, a custom timestamp for the measurement
            (integer containing seconds since epoch). If not given, the
            timestamp is obtained from the last modification time.
            delimiter: Column delimiter
            firstcoltime: If true, the first column is the treated as the
            sample time.

        """
        if os.path.exists(mfn):
            raise ValueError(f'File {mfn} already exist.')
        if timestamp is None:
            timestamp = int(time.time())

        if data.ndim != 2:
            data = data[:, np.newaxis]

        nchannels = data.shape[1]
        if nchannels != sensitivity.shape[0]:
            raise ValueError(
                f'Invalid sensitivity length given. Should be: {nchannels}')

        with h5.File(mfn, 'w') as hf:
            hf.attrs['samplerate'] = samplerate
            hf.attrs['sensitivity'] = sensitivity
            hf.attrs['time'] = timestamp
            hf.attrs['blocksize'] = 1
            hf.attrs['nchannels'] = nchannels
            ad = hf.create_dataset('audio', (1, data.shape[0], data.shape[1]),
                                   dtype=data.dtype,
                                   maxshape=(1, data.shape[0], data.shape[1]),
                                   compression='gzip')
            ad[0] = data
        return Measurement(mfn)
