import sys
include "config.pxi"
cimport cython
from .lasp_daqconfig import DeviceInfo
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cpython.ref cimport PyObject,Py_INCREF, Py_DECREF

# cdef extern from "lasp_worker.h":

cdef extern from "RtAudio.h" nogil:
    ctypedef unsigned long RtAudioStreamStatus
    RtAudioStreamStatus RTAUDIO_INPUT_OVERFLOW
    RtAudioStreamStatus RTAUDIO_OUTPUT_UNDERFLOW

    cdef cppclass RtAudioError:
        ctypedef enum Type:
            WARNING
            DEBUG_WARNING
            UNSPECIFIED
            NO_DEVICES_FOUND
            INVALID_DEVICE
            MEMORY_ERROR
            INVALID_PARAMETER
            INVALID_USE
            DRIVER_ERROR
            SYSTEM_ERROR
            THREAD_ERROR

    ctypedef unsigned long RtAudioStreamFlags
    RtAudioStreamFlags RTAUDIO_NONINTERLEAVED
    RtAudioStreamFlags RTAUDIO_MINIMIZE_LATENCY
    RtAudioStreamFlags RTAUDIO_HOG_DEVICE
    RtAudioStreamFlags RTAUDIO_SCHEDULE_REALTIME
    RtAudioStreamFlags RTAUDIO_ALSA_USE_DEFAULT
    RtAudioStreamFlags RTAUDIO_JACK_DONT_CONNECT

    ctypedef unsigned long RtAudioFormat
    RtAudioFormat RTAUDIO_SINT8
    RtAudioFormat RTAUDIO_SINT16
    RtAudioFormat RTAUDIO_SINT24
    RtAudioFormat RTAUDIO_SINT32
    RtAudioFormat RTAUDIO_FLOAT32
    RtAudioFormat RTAUDIO_FLOAT64

    ctypedef int (*RtAudioCallback)(void* outputBuffer,
                                    void* inputBuffer,
                                    unsigned int nFrames,
                                    double streamTime,
                                    RtAudioStreamStatus status,
                                    void* userData)

    ctypedef void (*RtAudioErrorCallback)(RtAudioError.Type _type,
                                          const string& errortxt)

    cdef cppclass cppRtAudio "RtAudio":
        cppclass DeviceInfo:
            bool probed
            string name
            unsigned int outputChannels
            unsigned int inputChannels
            unsigned int duplexChannels
            bool isDefaultOutput
            bool isDefaultInput
            vector[unsigned int] sampleRates
            unsigned int preferredSampleRate
            RtAudioFormat nativeFormats

        cppclass StreamOptions:
            RtAudioStreamFlags flags
            unsigned int numberOfBuffers
            string streamName
            int priority
        cppclass StreamParameters:
            unsigned int deviceId
            unsigned int nChannels
            unsigned int firstChannel

        RtAudio() except +
        # ~RtAudio() Destructors should not be listed
        unsigned int getDeviceCount()
        DeviceInfo getDeviceInfo(unsigned int device)
        unsigned int getDefaultOutputDevice()
        unsigned int getDefaultInputDevice()
        void openStream(StreamParameters* outputParameters,
                        StreamParameters* intputParameters,
                        RtAudioFormat _format,
                        unsigned int sampleRate,
                        unsigned int* bufferFrames,
                        RtAudioCallback callback,
                        void* userData,
                        void* StreamOptions,
                        RtAudioErrorCallback) except +
        void closeStream()
        void startStream() except +
        void stopStream() except +
        void abortStream() except +
        bool isStreamOpen()
        bool isStreamRunning()
        double getStreamTime()
        void setStreamTime(double) except +
        long getStreamLatency()
        unsigned int getStreamSampleRate()
        void showWarnings(bool value)
    
_formats_strkey = {
        '8-bit integers': (RTAUDIO_SINT8,   1, np.int8),
        '16-bit integers': (RTAUDIO_SINT16, 2, np.int16),
        '24-bit integers': (RTAUDIO_SINT24, 3),
        '32-bit integers': (RTAUDIO_SINT32, 4, np.int32),
        '32-bit floats': (RTAUDIO_FLOAT32,  4, np.float32),
        '64-bit floats': (RTAUDIO_FLOAT64,  8, np.float64),
}
_formats_rtkey = {
        RTAUDIO_SINT8: ('8-bit integers', 1, cnp.NPY_INT8),
        RTAUDIO_SINT16: ('16-bit integers',2, cnp.NPY_INT16),
        RTAUDIO_SINT24: ('24-bit integers',3),
        RTAUDIO_SINT32: ('32-bit integers',4, cnp.NPY_INT32),
        RTAUDIO_FLOAT32: ('32-bit floats',  4, cnp.NPY_FLOAT32),
        RTAUDIO_FLOAT64: ('64-bit floats',  8, cnp.NPY_FLOAT64),
}

def get_numpy_dtype_from_format_string(format_string):
    return _formats_strkey[format_string][-1]
def get_sampwidth_from_format_string(format_string):
    return _formats_strkey[format_string][-2]

ctypedef struct _Stream:
    PyObject* pyCallback
    RtAudioFormat sampleformat
    cppRtAudio.StreamParameters inputParams
    cppRtAudio.StreamParameters outputParams
    # These boolean values tell us whether the structs above here are
    # initialized and contain valid data
    bool hasInput
    bool hasOutput
    unsigned int bufferFrames



# It took me quite a long time to fully understand Cython's idiosyncrasies
# concerning C(++) callbacks, the GIL and passing Python objects as pointers
# into C(++) functions. But finally, here it is!
cdef object fromBufferToNPYNoCopy(
                             cnp.NPY_TYPES buffer_format_type,
                             void* buf,
                             size_t nchannels,
                             size_t nframes):
    cdef cnp.npy_intp[2] dims = [nframes, nchannels]
 
    # Interleaved data is C-style contiguous. Therefore, we can directly use
    # SimpleNewFromData()
    array = cnp.PyArray_SimpleNewFromData(2, &dims[0], buffer_format_type,
            buf)

    return array


cdef void fromNPYToBuffer(cnp.ndarray arr,
                          void* buf):
    """
    Copy a Python numpy array over to a buffer
    No checks, just memcpy! Careful!
    """
    memcpy(buf, arr.data, arr.size*arr.itemsize)


cdef int audioCallback(void* outputbuffer,
                       void* inputbuffer,
                       unsigned int nFrames,
                       double streamTime,
                       RtAudioStreamStatus status,
                       void* userData) nogil:
    """
    Calls the Python callback function and converts data

    """
    cdef:
        int rval = 0
        cnp.NPY_TYPES npy_format


    with gil:
        if status == RTAUDIO_INPUT_OVERFLOW:
            print('Input overflow.')
            return 0
        elif status == RTAUDIO_OUTPUT_UNDERFLOW:
            print('Output underflow.')
            return 0
        else:
            pass
        stream = <_Stream*>(userData)
        callback = <object> stream[0].pyCallback
        
        # Obtain stream information
        npy_input = None
        npy_output = None
        if stream.hasInput: 
            try:
                assert inputbuffer != NULL
                npy_format = _formats_rtkey[stream.sampleformat][2]
                npy_input = fromBufferToNPYNoCopy(
                        npy_format,
                        inputbuffer,
                        stream[0].inputParams.nChannels,
                        nFrames)

            except Exception as e:
                print('exception in cython callback for input: ', str(e))
                return 1

        if stream[0].hasOutput:
            try:
                assert outputbuffer != NULL
                npy_format = _formats_rtkey[stream[0].sampleformat][2]
                npy_output = fromBufferToNPYNoCopy(
                        npy_format,
                        outputbuffer,
                        stream[0].outputParams.nChannels,
                        nFrames)

            except Exception as e:
                print('exception in Cython callback for output: ', str(e))
                return 1
        try:
            rval = callback(npy_input,
                                     npy_output,
                                     nFrames,
                                     streamTime)
        except Exception as e:
            print('Exception in Cython callback: ', str(e))
            return 1

        return rval

cdef void errorCallback(RtAudioError.Type _type,const string& errortxt) nogil:
    with gil:
        print('Error callback called: %s', errortxt)




cdef class RtAudio:
    cdef:
        cppRtAudio _rtaudio
        _Stream* _stream

    def __cinit__(self):
        self._stream = NULL

    def __dealloc__(self):
        if self._stream is not NULL:
            print('Force closing stream')
            self._rtaudio.closeStream()

    cpdef unsigned int getDeviceCount(self):
        return self._rtaudio.getDeviceCount()

    cpdef unsigned int getDefaultOutputDevice(self):
        return self._rtaudio.getDefaultOutputDevice()

    cpdef unsigned int getDefaultInputDevice(self):
        return self._rtaudio.getDefaultInputDevice()

    def getDeviceInfo(self, unsigned int device):
        """
        Return device information of the current device
        """
        cdef cppRtAudio.DeviceInfo devinfo = self._rtaudio.getDeviceInfo(device)
        sampleformats = []
        nf = devinfo.nativeFormats
        for format_ in [ RTAUDIO_SINT8, RTAUDIO_SINT16, RTAUDIO_SINT24,
                RTAUDIO_SINT32, RTAUDIO_FLOAT32, RTAUDIO_FLOAT64]:
            if nf & format_:
                sampleformats.append(_formats_rtkey[format_][0])
        return DeviceInfo(
                index = device,
                probed = devinfo.probed,
                name = devinfo.name.decode('utf-8'),
                outputchannels = devinfo.outputChannels,
                inputchannels = devinfo.inputChannels,
                duplexchannels = devinfo.duplexChannels,
                samplerates = devinfo.sampleRates,
                sampleformats = sampleformats,
                prefsamplerate = devinfo.preferredSampleRate)

    @cython.nonecheck(True)
    def openStream(self,object outputParams,
                        object inputParams,
                        str sampleformat,
                        unsigned int sampleRate,
                        unsigned int bufferFrames,
                        object pyCallback,
                        object options = None,
                        object pyErrorCallback = None):
        """
        Opening a stream with specified parameters

        Args:
            outputParams: dictionary of stream outputParameters, set to None
            if no outputPararms are specified
            inputParams: dictionary of stream inputParameters, set to None
            if no inputPararms are specified
            sampleRate: desired sample rate.
            bufferFrames: the amount of frames in a callback buffer
            callback: callback to call. Note: this callback is called on a
            different thread!
            options: A dictionary of optional additional stream options
            errorCallback: client-defined function that will be invoked when an
            error has occured.

        Returns: None
        """
        if self._stream is not NULL:
            raise RuntimeError('Stream is already opened.')

        cdef cppRtAudio.StreamParameters *rtOutputParams_ptr = NULL
        cdef cppRtAudio.StreamParameters *rtInputParams_ptr = NULL

        cdef cppRtAudio.StreamOptions streamoptions
        streamoptions.flags = RTAUDIO_HOG_DEVICE
        streamoptions.numberOfBuffers = 4

        self._stream = <_Stream*> malloc(sizeof(_Stream))
        self._stream.hasInput = False
        self._stream.hasInput = False
        if self._stream is NULL:
            raise MemoryError()

        self._stream[0].pyCallback = <PyObject*> pyCallback
        Py_INCREF(pyCallback)
        self._stream[0].sampleformat = _formats_strkey[sampleformat][0]

        if outputParams is not None:
            rtOutputParams_ptr = &self._stream.outputParams
            rtOutputParams_ptr.deviceId = outputParams['deviceid']
            rtOutputParams_ptr.nChannels = outputParams['nchannels']
            rtOutputParams_ptr.firstChannel = outputParams['firstchannel']
            self._stream[0].hasOutput = True

        if inputParams is not None:
            rtInputParams_ptr = &self._stream.inputParams
            rtInputParams_ptr.deviceId = inputParams['deviceid']
            rtInputParams_ptr.nChannels = inputParams['nchannels']
            rtInputParams_ptr.firstChannel = inputParams['firstchannel']
            self._stream[0].hasInput = True

        try:
            self._stream.bufferFrames = bufferFrames
            self._rtaudio.openStream(rtOutputParams_ptr,
                                     rtInputParams_ptr,
                                     _formats_strkey[sampleformat][0],
                                     sampleRate,
                                     &self._stream[0].bufferFrames,
                                     audioCallback,
                                     <void*> self._stream,
                                     &streamoptions, # Stream options
                                     errorCallback  # Error callback
                                     ) 
        except Exception as e:
            print('Exception occured in stream opening: ', str(e))
            self._stream = NULL
            free(self._stream)
            Py_INCREF(pyCallback)
            raise

        return self._stream.bufferFrames

    def startStream(self):
        self._rtaudio.startStream()

    def stopStream(self):
        if self._stream is NULL:
            raise RuntimeError('Stream is not opened')
        self._rtaudio.stopStream()

    def closeStream(self):
        if self._stream is NULL:
            raise RuntimeError('Stream is not opened')
        # Closing stream
        self._rtaudio.closeStream()
        Py_DECREF(<object> self._stream[0].pyCallback)
        free(self._stream)
        self._stream = NULL

    def abortStream(self):
        if self._stream is NULL:
            raise RuntimeError('Stream is not opened')
        self._rtaudio.abortStream()

    def isStreamOpen(self):
        return self._rtaudio.isStreamOpen()

    def isStreamRunning(self):
        return self._rtaudio.isStreamRunning()

    def getStreamTime(self):
        return self._rtaudio.getStreamTime()
    
    def setStreamTime(self, double time):
        return self._rtaudio.setStreamTime(time)

    
