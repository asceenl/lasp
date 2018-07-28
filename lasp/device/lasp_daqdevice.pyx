include "config.pxi"
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf, stderr, fprintf
import sys
import numpy as np
cimport numpy as cnp

__all__ = ['DAQDevice']

from libc.errno cimport EPIPE, EBADFD, ESTRPIPE


cdef extern from "alsa/asoundlib.h":
    int snd_card_get_longname(int index,char** name)
    int snd_card_get_name(int index,char** name)
    int snd_card_next(int* rcard)

    ctypedef struct snd_pcm_t
    ctypedef struct snd_pcm_info_t
    ctypedef struct snd_pcm_hw_params_t
    ctypedef enum snd_pcm_stream_t:
        SND_PCM_STREAM_PLAYBACK
        SND_PCM_STREAM_CAPTURE
    ctypedef enum snd_pcm_format_t:
        SND_PCM_FORMAT_S16_LE
        SND_PCM_FORMAT_S16_BE
        SND_PCM_FORMAT_U16_LE
        SND_PCM_FORMAT_U16_BE
        SND_PCM_FORMAT_S24_LE
        SND_PCM_FORMAT_S24_BE
        SND_PCM_FORMAT_U24_LE
        SND_PCM_FORMAT_U24_BE
        SND_PCM_FORMAT_S32_LE
        SND_PCM_FORMAT_S32_BE
        SND_PCM_FORMAT_U32_LE
        SND_PCM_FORMAT_U32_BE
        SND_PCM_FORMAT_S24_3LE
        SND_PCM_FORMAT_S24_3BE
        SND_PCM_FORMAT_U24_3LE
        SND_PCM_FORMAT_U24_3BE
        SND_PCM_FORMAT_S16
        SND_PCM_FORMAT_U16
        SND_PCM_FORMAT_S24
        SND_PCM_FORMAT_U24
    const char* snd_pcm_format_name (snd_pcm_format_t format)
    ctypedef enum snd_pcm_access_t:
        SND_PCM_ACCESS_RW_INTERLEAVED
    ctypedef unsigned long snd_pcm_uframes_t
    int snd_pcm_open(snd_pcm_t** pcm,char* name, snd_pcm_stream_t type, int mode)
    int snd_pcm_close(snd_pcm_t*)

    int snd_pcm_hw_params_set_access(snd_pcm_t*, snd_pcm_hw_params_t*, snd_pcm_access_t)
    void snd_pcm_hw_params_alloca(snd_pcm_hw_params_t**)
    int snd_pcm_hw_params_any(snd_pcm_t*, snd_pcm_hw_params_t* params)
    int snd_pcm_hw_params_current(snd_pcm_t*, snd_pcm_hw_params_t* params)
    int snd_pcm_hw_params_set_rate_resample(snd_pcm_t*, snd_pcm_hw_params_t*, int)
    int snd_pcm_hw_params_set_rate(snd_pcm_t*, snd_pcm_hw_params_t*,unsigned int val,int dir)
    int snd_pcm_hw_params_set_format(snd_pcm_t*, snd_pcm_hw_params_t*, snd_pcm_format_t)
    int snd_pcm_hw_params_set_channels(snd_pcm_t*, snd_pcm_hw_params_t*, unsigned val)
    int snd_pcm_hw_params_set_period_size_near(snd_pcm_t*,snd_pcm_hw_params_t*,
        snd_pcm_uframes_t*,int* dir)
    int snd_pcm_hw_params_get_period_size(snd_pcm_hw_params_t*,
        snd_pcm_uframes_t*,int* dir)
    int snd_pcm_info(snd_pcm_t*, snd_pcm_info_t*)
    void snd_pcm_info_alloca(snd_pcm_info_t**)
    int snd_pcm_info_get_card(snd_pcm_info_t*)

    int snd_pcm_drain(snd_pcm_t*)
    int snd_pcm_readi(snd_pcm_t*,void* buf,snd_pcm_uframes_t nframes)
    int snd_pcm_hw_params(snd_pcm_t*,snd_pcm_hw_params_t*)
    int snd_pcm_hw_params_test_rate(snd_pcm_t*, snd_pcm_hw_params_t*,
                                    unsigned int val,int dir)
    int snd_pcm_hw_params_test_format(snd_pcm_t*, snd_pcm_hw_params_t*,
                                      snd_pcm_format_t)
    int snd_pcm_hw_params_get_channels_max(snd_pcm_hw_params_t*,unsigned int*)
    int snd_device_name_hint(int card, const char* iface, void*** hints)
    char* snd_device_name_get_hint(void* hint, const char* id)
    int snd_device_name_free_hint(void**)
    char* snd_strerror(int rval)

# Check for these sample rates
check_rates = [8000, 44100, 48000, 96000, 19200]

# First value in tuple: number of significant bits
# Second value:         number of bits used in memory
# Third value:          S for signed, U for unsigned, L for little endian,
#                       and B for big endian.
check_formats = {SND_PCM_FORMAT_S16_LE: (16,16,'SL'),
                 SND_PCM_FORMAT_S16_BE: (16,16,'SB'),
                 SND_PCM_FORMAT_U16_LE: (16,16,'UL'),
                 SND_PCM_FORMAT_U16_BE: (16,16,'UB'),
                 SND_PCM_FORMAT_S24_LE: (24,32,'SL'),
                 SND_PCM_FORMAT_S24_BE: (24,32,'SB'),
                 SND_PCM_FORMAT_U24_LE: (24,32,'UL'),
                 SND_PCM_FORMAT_U24_BE: (24,32,'UB'),
                 SND_PCM_FORMAT_S32_LE: (32,32,'SL'),
                 SND_PCM_FORMAT_S32_BE: (32,32,'SB'),
                 SND_PCM_FORMAT_U32_LE: (32,32,'UL'),
                 SND_PCM_FORMAT_U32_BE: (32,32,'UB'),
                 SND_PCM_FORMAT_S24_3LE: (24,24,'SL'),
                 SND_PCM_FORMAT_S24_3BE: (24,24,'SB'),
                 SND_PCM_FORMAT_U24_3LE: (24,24,'UL'),
                 SND_PCM_FORMAT_U24_3BE: (24,24,'UB')}

devices_opened_card = [False, False, False, False, False, False]

cdef snd_pcm_t* open_device(char* name,
                            snd_pcm_stream_t streamtype):
    """
    Helper function to properly open the first device of a card
    """
    cdef snd_pcm_t* pcm
    # if name in devices_opened:
        # raise RuntimeError('Device %s is already opened.' %name)
    cdef int rval = snd_pcm_open(&pcm, name, streamtype, 0)
    if rval == 0:
        return pcm
    else:
        return NULL

cdef int close_device(snd_pcm_t* dev):
    rval = snd_pcm_close(dev)
    if rval:
        print('Error closing device')
    return rval

class DeviceInfo:
    """
    Will later be replaced by a dataclass. Storage container for a lot of
    device parameters.
    """
    def __repr__(self):
        rep = f"""Device name: {self.device_name}
        """
        return rep

def getDeviceInfo(char* device_name):
    """
    Open the PCM device for both capture and playback, extract the number of
    channels, the samplerates and encoding

    Args:
        cardindex: Card number of device, numbered by ALSA

    Returns:

    """
    cdef:
        snd_pcm_t* pcm
        snd_pcm_hw_params_t* hwparams
        snd_pcm_info_t* info
        int rval, cardindex
        unsigned max_input_channels=0, max_output_channels=0
        char* c_cardname
        char *c_cardlongname

    deviceinfo = DeviceInfo()
    deviceinfo.device_name = device_name.decode('ASCII')
    pcm = open_device(device_name, SND_PCM_STREAM_CAPTURE)
    if not pcm:
        raise RuntimeError('Unable to open device')

    snd_pcm_info_alloca(&info)
    rval = snd_pcm_info(pcm, info)
    if rval:
        snd_pcm_close(pcm)
        raise RuntimeError('Unable to obtain device info')

    cardindex = snd_pcm_info_get_card(info)
    cardname = ''
    cardlongname = ''
    if cardindex >= 0:
        snd_card_get_name(cardindex, &c_cardname)
        if c_cardname:
            cardname = c_cardname.decode('ASCII')
            printf('name: %s\n', c_cardname)
            free(c_cardname)

        rval = snd_card_get_longname(cardindex, &c_cardlongname)
        if c_cardlongname:
            printf('longname: %s\n', c_cardlongname)
            cardlongname = c_cardlongname.decode('ASCII')
            free(c_cardlongname)
    deviceinfo.cardname = cardname
    deviceinfo.cardlongname = cardlongname

    # Check hardware parameters
    snd_pcm_hw_params_alloca(&hwparams)

    # Nothing said about the return value of this function in the documentation
    snd_pcm_hw_params_any(pcm, hwparams)

    # Check available sample formats
    available_formats = []
    for format in check_formats.keys():
        rval = snd_pcm_hw_params_test_format(pcm, hwparams, format)
        if rval == 0:
            available_formats.append(check_formats[format])
    deviceinfo.available_formats = available_formats
    # # Restrict a configuration space to contain only real hardware rates.
    # rval = snd_pcm_hw_params_set_rate_resample(pcm, hwparams, 0)
    # if rval !=0:
    #     fprintf(stderr, 'Unable disable resampling rates')

    # Check available input sample rates
    available_input_rates = []
    for rate in check_rates:
        rval = snd_pcm_hw_params_test_rate(pcm, hwparams, rate, 0)
        if rval == 0:
            available_input_rates.append(rate)
    deviceinfo.available_input_rates = available_input_rates

    rval = snd_pcm_hw_params_get_channels_max(hwparams, &max_input_channels)
    if rval != 0:
        fprintf(stderr, "Could not obtain max input channels\n")
    deviceinfo.max_input_channels = max_input_channels

    # Close device
    rval = snd_pcm_close(pcm)
    if rval:
        fprintf(stderr, 'Unable to close pcm device.\n')

    deviceinfo.available_output_rates = []
    deviceinfo.max_output_channels = 0

    # ###############################################################
    # Open device for output
    pcm = open_device(device_name, SND_PCM_STREAM_CAPTURE)
    if pcm == NULL:
        # We are unable to open the device for playback, but we were able to
        # open in for capture. So this is a valid device.
        return deviceinfo

    snd_pcm_hw_params_any(pcm, hwparams)
    # Restrict a configuration space to contain only real hardware rates.
    # rval = snd_pcm_hw_params_set_rate_resample(pcm, hwparams, 0)
    # if rval != 0:
    #     fprintf(stderr, 'Unable disable resampling rates')

    # Check available input sample rates
    available_output_rates = []
    for rate in check_rates:
        rval = snd_pcm_hw_params_test_rate(pcm, hwparams, rate, 0)
        if rval == 0:
            available_output_rates.append(rate)
    deviceinfo.available_output_rates = available_output_rates
    rval = snd_pcm_hw_params_get_channels_max(hwparams, &max_output_channels)
    if rval != 0:
        fprintf(stderr, "Could not obtain max output channels")
    deviceinfo.max_output_channels = max_output_channels

    # Close device
    rval = close_device(pcm)
    if rval:
        fprintf(stderr, 'Unable to close pcm device %s.', device_name)

    return deviceinfo


def query_devices():
    """
    Returns a list of available DAQ devices, where each device is represented
    by a dictionary containing parameters of the device
    """

    devices = []

    cdef:
        # Start cardindex at -1, such that the first one is picked by
        # snd_card_next()
        int cardindex = -1, rval=0, i=0
        void** namehints_opaque
        char** namehints
        char* c_device_name

    rval = snd_device_name_hint(-1, "pcm", &namehints_opaque)
    if rval:
        raise RuntimeError('Could not obtain name hints for card %i.'
        %cardindex)

    namehints = <char**> namehints_opaque
    while namehints[i] != NULL:
        # printf('namehint[i]: %s\n', namehints[i])
        c_device_name = snd_device_name_get_hint(namehints[i], "NAME")
        c_descr = snd_device_name_get_hint(namehints[i], "DESC")
        if c_device_name:
            device_name = c_device_name.decode('ASCII')
            if c_descr:
                device_desc = c_descr.decode('ASCII')
                free(c_descr)
            else:
                device_desc = ''
            try:
                device = getDeviceInfo(c_device_name)
                printf('device name: %s\n', c_device_name)
                devices.append(device)
            except RuntimeError:
                pass
            free(c_device_name)

        i+=1

    snd_device_name_free_hint(namehints_opaque)

    return devices



cdef class DAQDevice:
    cdef:
        snd_pcm_t* pcm
        int device_index
        object device, config
        public snd_pcm_uframes_t blocksize
        object callback


    def __cinit(self):
        self.pcm = NULL

    def __init__(self, config, blocksize=2048):
        """
        Initialize the DAQ device

        Args:
            config: DAQConfiguration instance
            blocksize: Number of frames in a single acquisition block
            callback: callback used to send data frames to
        """

        self.config = config
        devices = query_devices()

        self.device = None
        for device in devices:
            if self.config.match(device):
                # To open the underlying PCM device
                device_name = device.device_name.encode('ASCII')
                self.device = device

        if self.device is None:
            raise RuntimeError(f'Device {self.config.name} is not available')
        # if devices_opened[device_index]:
            # raise RuntimeError(f'Device {self.config.name} is already opened')
        # print('device_name opened:', device_name)
        self.pcm = open_device(device_name, SND_PCM_STREAM_CAPTURE)

        # Device is opened. We are going to configure
        cdef:
           snd_pcm_hw_params_t* params
           int rval
           snd_pcm_format_t format_code
           snd_pcm_uframes_t period_size
        snd_pcm_hw_params_alloca(&params);

        # Fill it in with default values.
        snd_pcm_hw_params_any(self.pcm, params);

        # Set access interleaved
        rval = snd_pcm_hw_params_set_access(self.pcm, params,
         SND_PCM_ACCESS_RW_INTERLEAVED)
        if rval != 0:
            snd_pcm_close(self.pcm)
            raise RuntimeError('Could not set access mode to interleaved')

        # Set sampling frequency
        cdef unsigned int rate
        rate = device.available_input_rates[config.en_input_rate]
        # printf('Set sample rate: %i\n', rate)
        rval = snd_pcm_hw_params_set_rate(self.pcm,params, rate, 0)
        if rval != 0:
            snd_pcm_close(self.pcm)
            raise RuntimeError('Could not set input sampling frequency')

        # Set number of channels
        channels_max = max(self.channels_en)+1
        # print('channels_max:', channels_max)
        if channels_max > self.device.max_input_channels:
            snd_pcm_close(self.pcm)
            raise ValueError('Highest required channel is larger than available'
            ' channels.')
        rval = snd_pcm_hw_params_set_channels(self.pcm, params, channels_max)
        if rval != 0:
            snd_pcm_close(self.pcm)
            raise RuntimeError('Could not set input channels, highest required'
            ' input channel: %i.' %channels_max)

        # Find the format description
        format_descr = self.device.available_formats[config.en_format]
        # Obtain key from value of dictionary

        format_code = list(check_formats.keys())[list(check_formats.values()).index(format_descr)]
        # printf('Format code: %s\n', snd_pcm_format_name(format_code))

        # print(format)
        rval = snd_pcm_hw_params_set_format(self.pcm, params, format_code)
        if rval != 0:
            fprintf(stderr, "Could not set format: %s.", snd_strerror(rval))

        # Set period size
        cdef int dir = 0
        bytedepth = format_descr[1]//8
        # print('byte depth:', bytedepth)
        period_size = blocksize
        rval = snd_pcm_hw_params_set_period_size_near(self.pcm,
                                                      params,
                                                      &period_size, &dir)

        if rval != 0:
            snd_pcm_close(self.pcm)
            raise RuntimeError("Could not set period size: %s."
            %snd_strerror(rval))

        # Write the parameters to the driver
        rc = snd_pcm_hw_params(self.pcm, params);
        if (rc < 0):
            snd_pcm_close(self.pcm)
            raise RuntimeError('Could not set hw parameters: %s' %snd_strerror(rc))

        # Check the block size again, and store it
        snd_pcm_hw_params_get_period_size(params, &self.blocksize,
                                          &dir)
        # print('Period size:', self.blocksize)

    cdef object _getEmptyBuffer(self):
        """
        Return right size empty buffer
        """
        format_descr =  self.device.available_formats[self.config.en_format]
        LB = format_descr[2][1]
        assert LB == 'L', 'Only little-endian data format supported'
        if format_descr[1] == 16:
            dtype = np.int16
        elif format_descr[1] == 32:
            dtype = np.int32

        # interleaved data, order = C
        return np.zeros((self.blocksize,
                         max(self.channels_en)+1),
                          dtype=dtype, order='C')

    def read(self):
        cdef int rval = 0
        buf = self._getEmptyBuffer()
        # buf2 = self._getEmptyBuffer()
        cdef cnp.int16_t[:, ::1] bufv = buf
        rval = snd_pcm_readi(self.pcm,<void*> &bufv[0, 0], self.blocksize)
        # rval = 2048
        if rval > 0:
            # print('Samples obtained:' , rval)
            return buf[:rval, self.channels_en]

            # return buf
        elif rval == -EPIPE:
            raise RuntimeError('Error: buffer overrun: %s',
            snd_strerror(rval))
        elif rval == -EBADFD:
            raise RuntimeError('Error: could not read from DAQ Device: %s',
            snd_strerror(rval))
        elif rval == -ESTRPIPE:
            raise RuntimeError('Error: could not read from DAQ Device: %s',
            snd_strerror(rval))


    def __dealloc__(self):
        # printf("dealloc\n")
        cdef int rval
        if self.pcm:
            # print('Closing pcm')
            # snd_pcm_drain(self.pcm)
            rval = snd_pcm_close(self.pcm)
            # devices_opened[self.device_index] = False
            if rval != 0:
                fprintf(stderr, 'Unable to properly close device: %s\n',
                snd_strerror(rval))
            self.pcm = NULL

    @property
    def nchannels_all(self):
        return self.device.max_input_channels

    @property
    def channels_en(self):
        return self.config.en_input_channels

    @property
    def input_rate(self):
        return self.device.available_input_rates[self.config.en_input_rate]

    @property
    def channels(self):
        return [self.config.en_input_channels]

    cpdef bint isOpened(self):
        if self.pcm:
            return True
        else:
            return False
